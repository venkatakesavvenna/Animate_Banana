"""Provider-neutral message and result types.

Every backend speaks these types at its edges; the provider-specific shape
(OpenAI content arrays, Gemini parts, Anthropic blocks, HF chat templates)
is built inside the backend from them and never leaks out. Agents therefore
construct one `list[Message]` regardless of which backend the config picks.
"""
from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    """An image referenced by path.

    Path rather than bytes so the same Message can be cheaply hashed for the
    response cache and cheaply logged; backends load lazily at send time.
    """
    path: Path

    def read_bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def media_type(self) -> str:
        guessed, _ = mimetypes.guess_type(str(self.path))
        return guessed or "image/png"

    def to_base64(self) -> str:
        return base64.b64encode(self.read_bytes()).decode("ascii")

    def to_data_url(self) -> str:
        return f"data:{self.media_type()};base64,{self.to_base64()}"


@dataclass
class VideoPart:
    """A video referenced by path.

    Same path-not-bytes rationale as ImagePart. Note that unlike an image, a
    video's bytes ARE hashed into the response-cache fingerprint (see
    `base._fingerprint`): two cells can share a prompt and a source image and
    differ only in the video, and without hashing it they would collide on one
    cache entry and the second cell would be served the first one's verdict.
    """
    path: Path

    def read_bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def media_type(self) -> str:
        guessed, _ = mimetypes.guess_type(str(self.path))
        return guessed or "video/mp4"

    def to_base64(self) -> str:
        return base64.b64encode(self.read_bytes()).decode("ascii")

    def to_data_url(self) -> str:
        return f"data:{self.media_type()};base64,{self.to_base64()}"


Part = TextPart | ImagePart | VideoPart


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: list[Part]

    @classmethod
    def user(cls, text: str, images: list[Path] | None = None,
             videos: list[Path] | None = None) -> "Message":
        """Build a user turn. Images precede text, which is what the chat VLMs
        in this project were prompted with in benchmark/infer.py and edges.py.

        Videos sit between the two, so a prompt reading "the original diagram
        as a still image, followed by a video" describes the actual part order.
        That ordering is load-bearing for the video fidelity prompt.
        """
        parts: list[Part] = [ImagePart(Path(p)) for p in (images or [])]
        parts.extend(VideoPart(Path(p)) for p in (videos or []))
        parts.append(TextPart(text))
        return cls(role="user", content=parts)

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role="system", content=[TextPart(text)])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls(role="assistant", content=[TextPart(text)])

    def text(self) -> str:
        """Concatenated text of this message, ignoring images."""
        return "\n".join(p.text for p in self.content if isinstance(p, TextPart))

    def images(self) -> list[ImagePart]:
        return [p for p in self.content if isinstance(p, ImagePart)]

    def videos(self) -> list[VideoPart]:
        return [p for p in self.content if isinstance(p, VideoPart)]


@dataclass
class GenerationResult:
    """Outcome of one generation call.

    `error` is set instead of raising so a batch of samples can survive one
    bad response -- the same failure-isolation choice run_benchmark.py made
    per-batch. Callers must check `ok` before trusting `text`.
    """
    text: str
    raw: object = None
    model: str = ""
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    error: str | None = None
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None
