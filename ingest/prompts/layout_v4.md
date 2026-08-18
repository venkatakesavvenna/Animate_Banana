Generate a single self-contained HTML file (with inline CSS only) that visually replicates the layout of a given document image.

Constraints:
- The final render area must be exactly <WIDTH>×<HEIGHT> pixels.
- Use absolute positioning for all elements to enable precise layout control.
- All text must use inline styles matching the original’s apparent font family, size, weight, color, and line height as closely as possible. Use standard web-safe fonts (e.g., Arial, Times New Roman, Georgia, Verdana) if the original font is unclear.
- Preserve the number of columns, relative order of content blocks (text and images), and approximate spacing between elements.
- For each image in the document, insert an <img> placeholder with:
    - src="ImageID" (e.g., src="img_1.png")
    - width and height attributes set to the provided pixel dimensions
    - style="position: absolute;" (do NOT include left/top values—omit them entirely)
- Do NOT include any bounding boxes, debug outlines, comments, or layout metadata.
- Output only the raw HTML—no explanations, markdown, or extra text.

You will be given:
- The total document size: <WIDTH>×<HEIGHT>
- A list of image placeholders, each specified by: ImageID, width, height

Now generate the HTML for a document of size <WIDTH>×<HEIGHT> with the following image placeholders:
<IMAGE_PLACEHOLDERS>