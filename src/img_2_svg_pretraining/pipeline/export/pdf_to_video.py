import fitz  # PyMuPDF
import imageio
import numpy as np
import cv2  # Added back for resizing
import os

def convert_pdf_to_mp4(pdf_path: str, mp4_path: str, fps: int = 2):
    """
    Converts a multi-page PDF into an MP4 video using ImageIO (FFmpeg).
    Forces every frame to identically match the dimensions of the first frame.
    """
    print(f"Opening {pdf_path}...")
    doc = fitz.open(pdf_path)
    
    if len(doc) == 0:
        print("Error: The PDF is empty.")
        return

    # 1. Establish the universal target size from the very first frame
    first_page = doc.load_page(0)
    first_pix = first_page.get_pixmap(dpi=300, alpha=False)
    target_width = first_pix.width
    target_height = first_pix.height
    
    print(f"Master frame dimensions locked at: {target_width}x{target_height}")

    # 2. Initialize the ImageIO video writer
    writer = imageio.get_writer(
        mp4_path, 
        fps=fps, 
        codec='libx264', 
        format='FFMPEG',
        macro_block_size=2 
    )

    # 3. Iterate through pages and write frames
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300, alpha=False)
        
        # Convert raw bytes into a NumPy array
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        
        # CRITICAL FIX: If the PDF page expanded due to the magnifying glass 
        # going out of bounds, resize it to exactly match the target dimensions.
        if img_array.shape[1] != target_width or img_array.shape[0] != target_height:
            img_array = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Write the uniform frame to the video
        writer.append_data(img_array)
        print(f"Processed frame {page_num + 1}/{len(doc)}")

    # 4. Close the writer to finalize the MP4
    writer.close()
    doc.close()
    print(f"Success! Saved universally playable video to: {mp4_path}")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # HARDCODED CONFIGURATION
    # ---------------------------------------------------------
    input_pdf = "slide_bbox_new_export.pdf"
    frames_per_second = 2 
    
    name, _ = os.path.splitext(input_pdf)
    output_mp4 = f"{name}.mp4"
    
    try:
        convert_pdf_to_mp4(input_pdf, output_mp4, fps=frames_per_second)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_pdf}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")