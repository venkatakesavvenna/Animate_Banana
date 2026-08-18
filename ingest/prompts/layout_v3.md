Generate a single self-contained HTML file (with inline CSS only) that visually replicates the layout of a given document image.

Constraints:
- Final render area must be exactly <WIDTH>×<HEIGHT> pixels.
- Preserve all columns, text blocks, and image placements as in the original.
- Use absolute positioning for precise layout control.
- All text must use inline styles matching the original’s font family, size, weight, color, and line spacing as closely as possible (assume standard web-safe fonts if unspecified).
- For each image, you will be given: ImageID (e.g., img_1.png), width, height, and top-left coordinates (x, y). Render each as an <img> element with:
    - src="ImageID"
    - width and height in pixels (fixed)
    - position: absolute; left: x px; top: y px;
- Do NOT include any bounding box annotations, debug outlines, or layout metadata in the output.
- Output only the HTML—no explanations, comments, or markdown.

Now generate the HTML for a document of size <WIDTH>×<HEIGHT> with the following content:
<IMAGE_PLACEHOLDER>