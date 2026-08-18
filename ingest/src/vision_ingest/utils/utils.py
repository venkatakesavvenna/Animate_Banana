from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image
import cv2
import os
import logging
import json, re
from copy import deepcopy
import traceback
from decimal import Decimal


class GracefulShutdown(Exception):
    """Custom exception for graceful shutdown via signals."""
    pass

def load_cv2_pil(p: Path, return_cv2: bool = True, return_pil: bool = True, logger=None):
    img = cv2.imread(str(p))   # BGR image

    if img is None:
        if logger:
            logger.error(f"Failed to read image: {p}")
        raise RuntimeError(f"Unreadable image: {p}")

    cv2_img = img if return_cv2 else None

    if return_pil:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB")
    else:
        pil_img = None

    return cv2_img, pil_img

class FsyncFileHandler(logging.FileHandler):
    """
    Adding an os.fsync wrapper around logging.FileHandler so that even in aws fsx, logging wont be buffered.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()
        os.fsync(self.stream.fileno())

def get_logger(log_dir, name):

    # Create a logger
    # logger_name = name --> TODO this causes a lot of issue!! (something about logger being global and process-wide singletons)
    logger_name = f"{name}:{os.path.abspath(log_dir)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO) 

    logger.propagate = False # --> TODO verify what it does
    
    if not logger.handlers:
        # Create a file handler that writes log messages to the file
        # file_handler = logging.FileHandler(log_file)

        # Create a log file path using the given directory and name
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")

        file_handler = FsyncFileHandler(log_file)
        file_handler.setLevel(logging.INFO)  # Set the file handler's level to INFO

        # Create a console handler to also print logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set the console handler's level to INFO

        # Define a log formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Apply the formatter to the handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean(v) for v in obj]
    elif isinstance(obj, Decimal):
        return float(obj)  # or str(obj)
    return obj
    
# def process_image(img_path: Path, logger=None):
#     """Extract metadata + caption with internal error handling."""
#     try:
#         img_path = Path(img_path)
#         # Replace with real extraction logic
#         meta = f"Metadata for {img_path.name}"
#         caption = f"Caption for {img_path.name}"
#         return meta, caption
#     except Exception as e:
#         if logger:
#             logger.error(f"process_image om utils/utils.py failed for {img_path}: {e}")
#         return "", ""

    
# def get_image_metadata_caption(image_paths: List[Path], reader_pool: ThreadPoolExecutor, logger=None):
#     metadata = [None] * len(image_paths)
#     captions = [None] * len(image_paths)

#     futures = {reader_pool.submit(process_image, p, logger): idx
#                for idx, p in enumerate(image_paths)}

#     for future in as_completed(futures):
#         idx = futures[future]
#         meta, caption = future.result()   # always safe now
#         metadata[idx] = meta
#         captions[idx] = caption

#     return metadata, captions


# def get_image_sizes(
#     image_paths: List[str],
#     logger=None
# ) -> List[Tuple[Optional[int], Optional[int]]]:
#     """
#     Returns list of (width, height) tuples for given image paths.
#     If an image fails to load, (None, None) is returned for that entry.
#     """
#     results: List[Tuple[Optional[int], Optional[int]]] = []

#     for path in image_paths:
#         try:
#             with Image.open(path) as img:
#                 results.append(img.size)  # (width, height)
#         except Exception as e:
#             if logger:
#                 logger.error(f"Failed to read image size for {path}: {type(e).__name__}: {e}")
#             results.append((None, None))

#     return results