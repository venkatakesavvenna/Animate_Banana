"""AnimateBanana human-evaluation study platform.

Runtime code in this package MUST NOT import `img_2_svg_pretraining.pipeline`.
Stimuli reach the app through a frozen bundle built once, offline, by
`study.build.build_bundle` -- the one subtree allowed that import.

The boundary is not tidiness. `pipeline/inspector/compare.py::_paths_for`
mutates a shared config to resolve a lineage and never restores it; in a
read-only viewer that races into showing the wrong video, but in a study it
serves the wrong condition to a participant and the response is unrecoverable
-- nothing on disk records which video they actually saw. Resolving a lineage
inside a request handler is the failure this package is arranged to prevent.
"""

BUNDLE_SCHEMA_VERSION = 1
