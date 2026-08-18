from pydantic import BaseModel, ValidationError as PydanticValidationError
from typing import Callable, Dict, List, Literal
import re
import json

class VLMOutput(BaseModel):
    short_caption: str
    medium_caption: str
    long_caption: str
    visual_caption: str
    semantic_caption: str
    has_text: bool
    text_languages: List[str]
    ocr_type: Literal["none", "scene_text", "text-rich_document", "visually-rich_document"]
    domain: Literal[
        "people","indoor","outdoor","food","documents","screenshots","products",
        "vehicles","animals","art","medical_scientific","text_overlay","unknown"
    ]
    complexity: Literal["simple","moderate","complex"]
    caption_type: Literal["descriptive","conversational","transcription","label"]
    image_quality: Literal["low","medium","high"]
    safety_flag: Literal["safe","sensitive_content"]
    alignment_score: float
    misalignment_reason: str

def extract_json_str(response_text: str) -> str:
    """
    Extract JSON string from model response.
    Handles <think> blocks and various JSON formatting.
    """
    # Remove <think>...</think>
    think_end = response_text.rfind("</think>")
    if think_end != -1:
        response_text = response_text[think_end + len("</think>"):]

    # Try ```json code block
    matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL))
    if matches:
        return matches[-1].group(1)

    # Fallback: try to find any {...} block
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("No JSON found in model output")

def parse_and_validate(response_text: str) -> Dict:
    """
    Parse and validate model response against VLMOutput schema.
    
    Args:
        response_text: Raw text output from the model
        
    Returns:
        Dictionary with validated VLM output fields
        
    Raises:
        ValueError: If JSON extraction fails
        PydanticValidationError: If validation against schema fails
    """
    if response_text==None :
        raise ValueError(f"Validation failed: {str(e)}")

    return {"text": response_text}
    try:
        json_str = extract_json_str(response_text)
        pydantic_obj = VLMOutput.model_validate_json(json_str)
        required_dict = pydantic_obj.model_dump()
        return required_dict
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except PydanticValidationError as e:
        # Re-raise as PydanticValidationError for proper catching
        raise e
    except Exception as e:
        raise ValueError(f"Validation failed: {str(e)}")
