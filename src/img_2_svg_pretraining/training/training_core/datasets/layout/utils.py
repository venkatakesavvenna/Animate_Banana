import torch
from datasets import Dataset
import numpy as np
from PIL import Image , ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import matplotlib.colors as mcolors


def split_train_eval(ds: Dataset, split_ratio = 0.1, seed=42):

    n_train = int(len(ds)*split_ratio)
    n_eval = len(ds) - n_train

    g = torch.Generator().manual_seed(int(seed))
    train_ds, eval_ds = torch.utils.data.random_split(ds, [n_train,n_eval],generator=g)
    return train_ds, eval_ds
 
def get_paragraph_order(component):
    """
    Sorts the paragraphs based on the top-down and left-right order
    """
    tlbr = component
    tlbr_sorted_x = sorted(tlbr, key=lambda x: x[0]) # x1 -> sorts with this (top left most box)
    
    mean_width = sum([box[3] for box in tlbr_sorted_x]) / len(tlbr_sorted_x)
    mean_x = mean_width

    current_vert_line = tlbr_sorted_x[0][0]
    vert_lines = []
    temp_line = []

    for box in tlbr_sorted_x:
        if box[0] >= current_vert_line + (mean_x):
            vert_lines.append(temp_line)
            temp_line = [box]
            current_vert_line = box[0]
            continue
        temp_line.append(box)
    vert_lines.append(temp_line)

    for line in vert_lines:
        line.sort(key=lambda x: x[1])

    return vert_lines


def _polygon_to_points(polygon) -> np.ndarray:
    """
    Normalises a polygon to an (N, 2) float32 numpy array.

    Accepted input formats:
      - np.ndarray of shape (N, 2)
      - list of (x, y) tuples or 2-element lists
      - flat list/array  [x1, y1, x2, y2, ...]
    """
    arr = np.asarray(polygon, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr


def get_polygon_order(polygons: list) -> list[list]:
    """
    Sorts a list of polygons in reading order (left-to-right columns,
    top-to-bottom within each column), mirroring `get_paragraph_order`.

    Each polygon is normalised via `_polygon_to_points` so all input
    formats supported by `visualise_promptable_dataset` are accepted.

    Returns:
        A list of columns, where each column is a list of polygons
        sorted top-to-bottom.  The columns themselves are ordered
        left-to-right, so iterating ``for col in result: for poly in col``
        yields polygons in natural reading order.
    """
    if not polygons:
        return []

    # Compute axis-aligned bounding box for every polygon
    bboxes = []  # (min_x, min_y, max_x, max_y) per polygon
    for poly in polygons:
        pts = _polygon_to_points(poly)
        min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
        max_x, max_y = pts[:, 0].max(), pts[:, 1].max()
        bboxes.append((min_x, min_y, max_x, max_y))

    # Pair each polygon with its bbox then sort by min_x (left edge)
    paired = sorted(zip(bboxes, polygons), key=lambda t: t[0][0])

    # Use mean bbox width as the column-split threshold (same logic as
    # get_paragraph_order which uses mean box[3] as the x threshold)
    mean_width = sum(b[2] - b[0] for b, _ in paired) / len(paired)

    current_col_x = paired[0][0][0]
    columns: list[list] = []
    temp_col: list = []

    for bbox, poly in paired:
        if bbox[0] >= current_col_x + mean_width:
            columns.append(temp_col)
            temp_col = [poly]
            current_col_x = bbox[0]
        else:
            temp_col.append(poly)
    columns.append(temp_col)

    # Within each column sort top-to-bottom by min_y of bounding box
    def _poly_min_y(poly):
        return _polygon_to_points(poly)[:, 1].min()

    for col in columns:
        col.sort(key=_poly_min_y)

    return columns


# Visually distinct colours for up to N masks
_MASK_COLORS = [
    (255,  60,  60),   # red
    ( 60, 180, 255),   # sky-blue
    ( 60, 220,  60),   # green
    (255, 180,   0),   # amber
    (200,   0, 255),   # purple
    (  0, 220, 200),   # teal
    (255, 120,   0),   # orange
    (255,   0, 160),   # pink
    (160, 255,   0),   # lime
    (  0, 100, 255),   # blue
]

def visualise_promptable_dataset(
    copied_original_image: Image.Image,
    debug_path,
    mask_layouts=None,
    question: str = "",
    ret_str: str = "",
    alpha: float = 0.45,
    text_width_chars: int = 60,
    use_bboxes: bool = False,
    bboxes: list = None,
    bbox_format: str = "xywh",
    use_polygons: bool = False,
    polygons: list = None,
    entire_sample: dict = None,
):
    """
    Visualizes the promptable dataset by showing:
    1. Image with masks/bboxes/polygons overlaid (left side)
    2. Question and ret_str in a white text area (right side)
    3. Entire sample data at the bottom (if provided)

    All sizing is derived from the actual image dimensions so text is
    always legible regardless of input resolution.
    
    Args:
        use_bboxes: If True, uses bboxes instead of masks
        bboxes: List of bounding boxes (when use_bboxes=True)
        bbox_format: "xywh" or "xyxy" format for bboxes
        use_polygons: If True, uses polygons instead of masks/bboxes
        polygons: List of polygons (when use_polygons=True). Each polygon can be:
                  - List of tuples: [(x1, y1), (x2, y2), ...]
                  - List of lists: [[x1, y1], [x2, y2], ...]
                  - Flattened list: [x1, y1, x2, y2, ...]
                  - Numpy array of shape (N, 2)
        entire_sample: Optional dict containing the entire sample data to display
    """
    from textwrap import wrap
    import json

    debug_dir = Path(debug_path)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # -------- OVERLAY MASKS, BBOXES, OR POLYGONS ON IMAGE --------
    orig_np = np.array(copied_original_image).astype(np.float32)
    composite = orig_np.copy()

    if use_polygons and polygons is not None:
        # Draw polygons
        image_with_polygons = Image.fromarray(composite.astype(np.uint8))
        draw = ImageDraw.Draw(image_with_polygons)
        
        # Calculate font size for numbering based on image size
        label_font_size = max(16, int(min(image_with_polygons.width, image_with_polygons.height) * 0.02))
        try:
            label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_font_size)
        except Exception:
            label_font = ImageFont.load_default()
        
        for idx, polygon in enumerate(polygons):
            colour = _MASK_COLORS[idx % len(_MASK_COLORS)]
            
            # Convert polygon to proper format for PIL
            if isinstance(polygon, np.ndarray):
                if polygon.ndim == 1:
                    polygon = polygon.reshape(-1, 2)
                points = polygon.tolist()
            else:
                polygon_array = np.array(polygon)
                if polygon_array.ndim == 1:
                    polygon_array = polygon_array.reshape(-1, 2)
                points = polygon_array.tolist()
            
            # Flatten to list of coordinates for PIL
            flat_points = [coord for point in points for coord in point]
            
            # Draw polygon outline with thicker lines
            line_width = max(3, int(min(image_with_polygons.width, image_with_polygons.height) * 0.003))
            draw.polygon(flat_points, outline=colour, width=line_width)
            
            # Calculate centroid for label placement
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            cx = int(sum(x_coords) / len(x_coords))
            cy = int(sum(y_coords) / len(y_coords))
            
            # Add numbered label at centroid
            label_text = str(idx + 1)
            bbox_text = draw.textbbox((cx, cy), label_text, font=label_font)
            padding = max(4, int(label_font_size * 0.2))
            # Draw background rectangle
            draw.rectangle(
                [bbox_text[0] - padding, bbox_text[1] - padding, 
                 bbox_text[2] + padding, bbox_text[3] + padding],
                fill=(0, 0, 0)
            )
            # Draw the number
            draw.text((cx, cy), label_text, fill=(255, 255, 255), font=label_font)
        
        image_with_masks = image_with_polygons
    elif use_bboxes and bboxes is not None:
        # Draw bounding boxes
        image_with_boxes = Image.fromarray(composite.astype(np.uint8))
        draw = ImageDraw.Draw(image_with_boxes)
        
        # Calculate font size for numbering based on image size
        label_font_size = max(16, int(min(image_with_boxes.width, image_with_boxes.height) * 0.02))
        try:
            label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_font_size)
        except Exception:
            label_font = ImageFont.load_default()
        
        for idx, bbox in enumerate(bboxes):
            colour = _MASK_COLORS[idx % len(_MASK_COLORS)]
            
            if bbox_format == "xywh":
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
            else:  # xyxy
                x1, y1, x2, y2 = bbox
            
            # Draw rectangle with thicker lines for better visibility
            box_width = max(3, int(min(image_with_boxes.width, image_with_boxes.height) * 0.003))
            draw.rectangle([x1, y1, x2, y2], outline=colour, width=box_width)
            
            # Add numbered label at top-left corner of bbox
            label_text = str(idx + 1)
            # Get text bounding box for background
            bbox_text = draw.textbbox((x1, y1), label_text, font=label_font)
            padding = max(4, int(label_font_size * 0.2))
            # Draw background rectangle
            draw.rectangle(
                [bbox_text[0] - padding, bbox_text[1] - padding, 
                 bbox_text[2] + padding, bbox_text[3] + padding],
                fill=(0, 0, 0)
            )
            # Draw the number
            draw.text((x1, y1), label_text, fill=(255, 255, 255), font=label_font)
        
        image_with_masks = image_with_boxes
    else:
        # Draw masks (original behavior)
        mask_centroids = []  # Store centroids for numbering
        
        for idx, mask_layout in enumerate(mask_layouts):
            colour = _MASK_COLORS[idx % len(_MASK_COLORS)]
            colour_layer = np.zeros_like(orig_np, dtype=np.float32)
            colour_layer[..., 0] = colour[0]
            colour_layer[..., 1] = colour[1]
            colour_layer[..., 2] = colour[2]

            mask_np = mask_layout.detach().cpu().numpy()
            m = (mask_np > 0).astype(np.float32)[..., None]
            composite = composite * (1.0 - alpha * m) + colour_layer * (alpha * m)
            
            # Calculate centroid for label placement
            mask_2d = (mask_np > 0).astype(np.uint8)
            ys, xs = np.where(mask_2d > 0)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
            else:
                cx, cy = 20, 20 + idx * 30
            mask_centroids.append((cx, cy))

        composite = composite.clip(0, 255).astype(np.uint8)
        image_with_masks = Image.fromarray(composite)
        
        # Add numbered labels to masks
        draw = ImageDraw.Draw(image_with_masks)
        label_font_size = max(16, int(min(image_with_masks.width, image_with_masks.height) * 0.02))
        try:
            label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_font_size)
        except Exception:
            label_font = ImageFont.load_default()
        
        for idx, (cx, cy) in enumerate(mask_centroids):
            label_text = str(idx + 1)
            # Get text bounding box for background
            bbox_text = draw.textbbox((cx, cy), label_text, font=label_font)
            padding = max(4, int(label_font_size * 0.2))
            # Draw background rectangle
            draw.rectangle(
                [bbox_text[0] - padding, bbox_text[1] - padding, 
                 bbox_text[2] + padding, bbox_text[3] + padding],
                fill=(0, 0, 0)
            )
            # Draw the number
            draw.text((cx, cy), label_text, fill=(255, 255, 255), font=label_font)

    img_w, img_h = image_with_masks.size

    # -------- SCALE EVERYTHING OFF IMAGE SIZE --------
    # Text panel width: 45% of image width, min 300px
    text_area_width = max(300, int(img_w * 0.45))

    # Font size: ~2.2% of image height for title, ~1.8% for body, ~1.3% for sample data, with sane min/max
    font_size_title = int(max(18, min(60, img_h * 0.022)))
    font_size_body  = int(max(15, min(50, img_h * 0.018)))
    font_size_sample = int(max(12, min(40, img_h * 0.013)))  # Smaller font for entire sample

    # Padding & spacing scale with font size
    PADDING       = font_size_body * 2
    LINE_SPACING  = int(font_size_body * 0.45)
    LINE_SPACING_SAMPLE = int(font_size_sample * 0.4)  # Tighter line spacing for sample
    SECTION_GAP   = font_size_body * 2

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size_title)
        font_body  = ImageFont.truetype("DejaVuSans.ttf",      font_size_body)
        font_sample = ImageFont.truetype("DejaVuSans.ttf",     font_size_sample)
    except Exception:
        font_title = ImageFont.load_default()
        font_body  = ImageFont.load_default()
        font_sample = ImageFont.load_default()

    # -------- WORD WRAP --------
    # Estimate chars per line: usable width divided by avg char width (~0.55 * font_size)
    usable_px      = text_area_width - 2 * PADDING
    avg_char_px    = font_size_body * 0.55
    chars_per_line = max(20, int(usable_px / avg_char_px))

    question_lines = wrap(f"Question: {question}", width=chars_per_line)
    answer_lines   = wrap(f"Answer: {ret_str}",    width=chars_per_line)
    
    # Format entire sample if provided (with smaller font, more chars per line)
    sample_lines = []
    if entire_sample is not None:
        # Use more chars per line for sample data (smaller font)
        avg_char_px_sample = font_size_sample * 0.55
        chars_per_line_sample = max(30, int(usable_px / avg_char_px_sample))
        
        sample_json = json.dumps(entire_sample, indent=2, ensure_ascii=False)
        sample_lines = wrap(f"Entire Sample: {sample_json}", width=chars_per_line_sample)

    # -------- MEASURE TEXT HEIGHT --------
    probe_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def line_h(line, font):
        bb = probe_draw.textbbox((0, 0), line, font=font)
        return bb[3] - bb[1]

    total_text_h = PADDING * 2 + SECTION_GAP
    for line in question_lines:
        total_text_h += line_h(line, font_title) + LINE_SPACING
    for line in answer_lines:
        total_text_h += line_h(line, font_body) + LINE_SPACING
    if sample_lines:
        total_text_h += SECTION_GAP
        for line in sample_lines:
            total_text_h += line_h(line, font_sample) + LINE_SPACING_SAMPLE

    text_area_height = max(img_h, total_text_h)

    # -------- RENDER TEXT PANEL --------
    text_img = Image.new("RGB", (text_area_width, text_area_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(text_img)

    y = PADDING

    # Question (bold)
    for line in question_lines:
        draw.text((PADDING, y), line, fill=(15, 15, 15), font=font_title)
        y += line_h(line, font_title) + LINE_SPACING

    y += SECTION_GAP

    # Answer (regular)
    for line in answer_lines:
        draw.text((PADDING, y), line, fill=(40, 40, 40), font=font_body)
        y += line_h(line, font_body) + LINE_SPACING
    
    # Entire Sample (smaller font)
    if sample_lines:
        y += SECTION_GAP
        for line in sample_lines:
            draw.text((PADDING, y), line, fill=(60, 60, 60), font=font_sample)
            y += line_h(line, font_sample) + LINE_SPACING_SAMPLE

    # -------- MATCH HEIGHTS WITH WHITE PADDING (no distorting resize) --------
    final_h = max(img_h, text_area_height)

    if img_h < final_h:
        padded_img = Image.new("RGB", (img_w, final_h), color=(255, 255, 255))
        padded_img.paste(image_with_masks, (0, 0))
        image_with_masks = padded_img

    if text_area_height < final_h:
        padded_text = Image.new("RGB", (text_area_width, final_h), color=(255, 255, 255))
        padded_text.paste(text_img, (0, 0))
        text_img = padded_text

    # -------- COMBINE SIDE-BY-SIDE WITH DIVIDER --------
    divider_w   = max(2, int(img_w * 0.002))  # ~0.2% of image width
    combined_w  = image_with_masks.width + divider_w + text_img.width

    combined_img = Image.new("RGB", (combined_w, final_h), color=(180, 180, 180))
    combined_img.paste(image_with_masks, (0, 0))
    combined_img.paste(text_img, (image_with_masks.width + divider_w, 0))

    # -------- SAVE --------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outpath   = debug_dir / f"visual_{timestamp}.jpg"
    combined_img.save(str(outpath), quality=95)
    print(f"[debug] wrote promptable visualisation to {outpath}")


def visualise_input_dataset(
    copied_original_image: Image.Image,
    debug_path,
    mask_layouts,
    category_names,
    alpha: float = 0.45,
):
    """
    Produces a SINGLE combined image where every layout mask is painted in a
    distinct colour and labelled with its category name.

    Saved to:  <debug_path>/all_masks_combined_<timestamp>.jpg
    """

    debug_dir = Path(debug_path)
    debug_dir.mkdir(parents=True, exist_ok=True)

    orig_np = np.array(copied_original_image).astype(np.float32)
    composite = orig_np.copy()

    n = len(mask_layouts)
    label_positions = []

    for idx in range(n):
        colour = _MASK_COLORS[idx % len(_MASK_COLORS)]

        colour_layer = np.zeros_like(orig_np, dtype=np.float32)
        colour_layer[..., 0] = colour[0]
        colour_layer[..., 1] = colour[1]
        colour_layer[..., 2] = colour[2]

        mask_np = mask_layouts[idx].detach().cpu().numpy()
        mask = (mask_np > 0).astype(np.float32)
        m = mask[..., None]

        # Blend mask
        composite = composite * (1.0 - alpha * m) + colour_layer * (alpha * m)

        # Compute centroid for label placement
        ys, xs = np.where(mask > 0)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            cx, cy = 10, 20 + idx * 25

        label_positions.append((cx, cy, category_names[idx], colour))

    composite = composite.clip(0, 255).astype(np.uint8)
    result_img = Image.fromarray(composite)

    # -------- DRAW LABELS --------
    draw = ImageDraw.Draw(result_img)

    # Dynamically scale font based on image size
    img_w, img_h = result_img.size
    font_size = max(18, int(min(img_w, img_h) * 0.025))

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for (cx, cy, name, colour) in label_positions:

        tx = cx + 5
        ty = cy - 5

        # Compute text bounding box
        bbox = draw.textbbox((tx, ty), name, font=font)
        x0, y0, x1, y1 = bbox

        padding = 6

        # Background rectangle (solid black for maximum contrast)
        draw.rectangle(
            [x0 - padding, y0 - padding, x1 + padding, y1 + padding],
            fill=(0, 0, 0)
        )

        # White text for readability
        draw.text(
            (tx, ty),
            name,
            fill=(255, 255, 255),
            font=font
        )

    # -------- SAVE OUTPUT --------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outpath = debug_dir / f"all_masks_combined_{timestamp}.jpg"

    result_img.save(str(outpath))
    print(f"[debug] wrote combined visualisation to {outpath}")



def save_image_with_bboxes(
    image_path: str,
    bboxes: list,
    format_type: str,   # "xyxy" or "xywh"
    dir_path: str,
    box_color: str = "red",
    box_width: int = 3
):
    """
    Draw bounding boxes on an image and save it with timestamp name.

    Args:
        image_path (str): Path to input image.
        bboxes (list): List of bounding boxes.
        format_type (str): 'xyxy' or 'xywh'.
        dir_path (str): Directory to save output image.
        box_color (str): Color of bounding box.
        box_width (int): Line thickness.
    """

    # Validate format
    if format_type not in ["xyxy", "xywh"]:
        raise ValueError("format_type must be either 'xyxy' or 'xywh'")

    # Create output directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Open image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw boxes
    for bbox in bboxes:
        if format_type == "xyxy":
            x1, y1, x2, y2 = bbox
        else:  # xywh
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

    # Generate timestamp filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(dir_path, f"{timestamp}.png")

    # Save image
    image.save(output_path)

    return output_path



def draw_mask_overlay_and_box(orig_image_np: np.ndarray, mask_tensor:torch.Tensor, box=None, outpath=None, alpha=0.5):
    """
    orig_image_np: HxWx3 uint8
    mask_np: HxW (bool/0-1/0-255). Non-zero is treated as mask=1
    box: (x1, y1, x2, y2) or None
    If outpath is provided, saves the OVERLAY image figure there. Only the overlay is shown.
    """


    mask_np = mask_tensor.detach().cpu().numpy()

    mask = (mask_np.astype(np.uint8) > 0).astype(np.uint8)

    # red color layer
    red = np.zeros_like(orig_image_np, dtype=np.uint8)
    red[..., 0] = 255

    # blend only where mask==1
    m = mask[..., None].astype(np.float32)  # HxWx1
    overlay = (orig_image_np.astype(np.float32) * (1.0 - alpha * m) +
               red.astype(np.float32) * (alpha * m)).astype(np.uint8)

    # single image (no subplots)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    ax = plt.gca()
    ax.set_axis_off()

    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)  # lime
        overlay = np.array(img)

    if outpath:
        Image.fromarray(overlay).save(outpath)  # PNG default = lossless

# def visualise_input_dataset(copied_original_image:Image.Image, debug_path, mask_layouts, category_names):
#     debug_dir = Path(debug_path)
#     debug_dir.mkdir(parents= True, exist_ok= True)

#     # Convert PIL → numpy
#     orig_np = np.array(copied_original_image)

#     for idx in range(len(mask_layouts)):
#         outpath = debug_dir / f"sample_visual_{category_names[idx]}_{idx}.jpg"
#         draw_mask_overlay_and_box(orig_np, mask_layouts[idx], outpath=str(outpath))
#         print(f"[debug] wrote visualization to {outpath}")

