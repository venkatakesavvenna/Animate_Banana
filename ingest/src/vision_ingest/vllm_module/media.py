"""
Image-input validation — the single place every image path is checked before it
is allowed to reach any backend.

Why this module exists (v1.7)
-----------------------------
Before this, an image path went to the engine unchecked and each backend failed
*differently* and *quietly*:

  - online  — the path was pasted into a `file://` URL with no encoding, so any
              path containing `#`, `?` or `%` was silently truncated/rewritten by
              urllib3's parser inside vLLM. A non-existent file, an unreadable
              file, or a path outside `--allowed-local-media-path` came back as
              an HTTP 400 whose body we discarded, so the stage saw only
              `output=None` -> "retry" -> "failed after 3 attempts", with no
              indication of *why*.
  - offline — `cv2.imread` returns None for a missing/corrupt file, which
              `load_cv2_pil` turns into a generic RuntimeError; oversized images
              were accepted with no limit at all (vLLM only enforces
              VLLM_MAX_IMAGE_PIXELS on the server side), so the two backends
              disagreed on what is even a legal input.

Every check here raises `ImageValidationError` with the exact reason and the
`repr()` of the offending path (so encoding problems — mojibake, NFC/NFD
mismatch, literal "\\u2014" escapes, surrogates from an undecodable filename —
are visible in one log line instead of being guessed at).

Nothing here ever "fixes up" a path. A path is either valid, or the item fails
loudly with a reason. Silent normalisation is exactly how a 100M-image run ends
up with a fraction of its outputs generated from the wrong (or no) image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

from PIL import Image

# NOTE: PIL's own decompression-bomb limit is deliberately left ALONE. Raising
# or disabling it here would apply process-wide — including to project code that
# opens images itself — and would trade an explicit error for a possible OOM.
# Its hard threshold (2x Image.MAX_IMAGE_PIXELS, ~179M px) happens to coincide
# with vLLM's own ceiling below; when it fires first we translate it into the
# same oversize message rather than a generic decode failure.

# Mirrors vllm/envs.py's VLLM_MAX_IMAGE_PIXELS default (PIL's 2x decompression
# bomb threshold, ~179M pixels / ~680 MB as RGB). The server enforces this on
# the online path whether we like it or not, so the offline path enforces the
# same number to keep "what counts as a legal image" backend-independent.
DEFAULT_MAX_IMAGE_PIXELS = int(os.getenv("VLLM_MAX_IMAGE_PIXELS", "178956970"))


class ImageValidationError(ValueError):
    """One image failed validation. Always carries the reason and the path."""


@dataclass(frozen=True)
class ImageInfo:
    """What validation learned about one image. `width`/`height` are None when
    the size was not read (offline path — the loader reads the pixels anyway)."""
    path: str
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def pixels(self) -> Optional[int]:
        if self.width is None or self.height is None:
            return None
        return self.width * self.height


@dataclass(frozen=True)
class ImagePolicy:
    """
    Rules every image path must satisfy, resolved once at startup.

    allowed_media_roots
        Directories images may live under. On the online backend this MUST match
        `vllm serve --allowed-local-media-path`, because the server rejects
        anything outside it; validating here turns that server-side 400 into a
        precise, local, pre-submit error. Empty tuple = no root restriction
        (offline backends only — `online_policy()` refuses to build one).
    max_pixels
        Hard upper bound on `width * height`. Anything above fails as a terminal
        item error rather than crashing a worker or being silently accepted by
        one backend and rejected by the other. 0 disables the check.
    processor_max_pixels
        Informational only: the model's own image processor downscales anything
        above this (e.g. Qwen2VLImageProcessorFast `size.longest_edge`). Used to
        report how many inputs the model will resize, so quality loss from
        downscaling is a logged number instead of an invisible default.
    read_header
        Whether to open the file and read its header. True online (nothing else
        reads the file locally, so this is the only chance to catch a corrupt or
        oversized image before the server does); False offline (the loader
        decodes the whole image immediately afterwards anyway).
    """
    allowed_media_roots: Tuple[str, ...] = ()
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS
    processor_max_pixels: Optional[int] = None
    read_header: bool = False

    # ---------------- constructors ----------------

    @staticmethod
    def normalise_roots(roots) -> Tuple[str, ...]:
        """Accept a str or an iterable of str; return absolute, real paths."""
        if roots is None:
            return ()
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        out = []
        for r in roots:
            if not r:
                continue
            real = os.path.realpath(str(r))
            if not os.path.isdir(real):
                raise ValueError(
                    f"allowed media root does not exist or is not a directory: {r!r}"
                )
            out.append(real)
        return tuple(out)

    def describe(self) -> dict:
        return {
            "allowed_media_roots": list(self.allowed_media_roots) or "unrestricted",
            "max_pixels": self.max_pixels or "unlimited",
            "processor_downscales_above_pixels": self.processor_max_pixels,
            "reads_image_header": self.read_header,
        }


def _fail(path: str, reason: str) -> "ImageValidationError":
    # repr() the path deliberately: it is the only way a bad encoding
    # (mojibake, a literal "—", an undecodable-filename surrogate) shows up
    # as itself in a log instead of as a look-alike character.
    return ImageValidationError(f"{reason}: {path!r}")


def validate_image_path(path, policy: ImagePolicy) -> ImageInfo:
    """
    Check one image path against `policy`. Returns ImageInfo on success, raises
    ImageValidationError (never anything else) on any failure.
    """
    if not isinstance(path, str):
        raise _fail(str(path), f"image path must be a str, got {type(path).__name__}")
    if not path:
        raise _fail(path, "image path is empty")
    if not os.path.isabs(path):
        # Relative paths are resolved against the *worker's* cwd on the offline
        # path and are meaningless in a file:// URL — never guess a base dir.
        raise _fail(path, "image path must be absolute")
    # A filename that could not be decoded with the filesystem encoding comes
    # back from os.listdir with lone surrogates; those blow up much later (in
    # json.dumps or in the HTTP encoder) with an unrelated-looking error.
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as e:
        raise _fail(path, f"image path is not valid UTF-8 ({e})") from None

    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise _fail(path, "image file does not exist") from None
    except OSError as e:
        raise _fail(path, f"image file is not accessible ({e.strerror})") from None
    if not os.path.isfile(path):
        raise _fail(path, "image path is not a regular file")
    if st.st_size == 0:
        raise _fail(path, "image file is empty (0 bytes)")

    if policy.allowed_media_roots:
        real = os.path.realpath(path)
        if not any(
            real == root or real.startswith(root + os.sep)
            for root in policy.allowed_media_roots
        ):
            raise _fail(
                path,
                "image path is outside the allowed media roots "
                f"{list(policy.allowed_media_roots)} (resolved to {real!r}); the "
                "vllm serve --allowed-local-media-path would reject it",
            )

    if not policy.read_header:
        return ImageInfo(path)

    try:
        with Image.open(path) as im:
            width, height = im.size
    except Image.DecompressionBombError as e:
        # PIL's own ceiling tripped before ours. Same class of problem — report
        # it as oversize, not as "corrupt", so the log says what to do about it.
        raise _fail(
            path,
            f"image exceeds PIL's decompression-bomb limit ({e}); it is far above "
            f"the {policy.max_pixels}-pixel ceiling and cannot be processed",
        ) from None
    except Exception as e:
        # UnidentifiedImageError / OSError / anything PIL throws on a corrupt or
        # non-image file. The alternative is the server rejecting it with a 400
        # we would then retry three times for no reason.
        raise _fail(path, f"image cannot be decoded ({type(e).__name__}: {e})") from None

    check_pixels(path, width, height, policy)
    return ImageInfo(path, width, height)


def check_pixels(path: str, width: int, height: int, policy: ImagePolicy) -> None:
    """Enforce the pixel ceiling. Split out so the offline loader can call it
    with the size it already read instead of re-opening the file."""
    if policy.max_pixels and width * height > policy.max_pixels:
        raise _fail(
            path,
            f"image is {width}x{height} = {width * height} pixels, above the "
            f"{policy.max_pixels}-pixel limit (VLLM_MAX_IMAGE_PIXELS); vllm serve "
            "rejects these outright, so the offline backends reject them too",
        )


def validate_image_paths(paths: Sequence[str], policy: ImagePolicy) -> List[ImageInfo]:
    """Validate every path in a prompt. Fails on the first bad one — a prompt is
    all-or-nothing, there is no such thing as a partially-imaged request."""
    return [validate_image_path(p, policy) for p in paths]


def to_file_url(path: str) -> str:
    """
    Build the `file://` URL for an already-validated absolute path.

    `quote()` is not optional. vLLM parses the URL with `urllib3.util.parse_url`
    and then un-quotes it with `urllib.request.url2pathname`, so a raw path
    containing `#` or `?` is truncated at that character, and a literal `%`
    sequence is decoded into something else entirely — in both cases the server
    silently reads a *different* file, or reports one that "does not exist".
    Percent-encoding here makes the round-trip exact for every byte.
    """
    return "file://" + quote(path)


def summarise_downscaling(infos: Iterable[ImageInfo], policy: ImagePolicy) -> int:
    """Count how many of these images the model's own processor will downscale.
    Returns 0 when the processor's limit is unknown."""
    if not policy.processor_max_pixels:
        return 0
    return sum(
        1 for i in infos
        if i.pixels is not None and i.pixels > policy.processor_max_pixels
    )
