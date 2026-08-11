"""Human-in-the-loop correction layer for pipeline Stage 1.

Lives inside the pipeline package (not beside it, like `annotation_tool/`)
because it reads and writes `CachePaths`-addressed artifacts directly.
"""
