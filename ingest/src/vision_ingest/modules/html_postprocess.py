import json
import re
from pathlib import Path
from typing import Union, Dict, Any


def extract_html_block(
    json_source: Union[str, Path, Dict[str, Any]],
    output_html_path: Union[str, Path],
) -> Path:
    """
    Parse a JSON (or dict), extract the content between ```html ... ``` fences
    from json['vlm_output']['text'], and save it to an HTML file.

    Args:
        json_source: Either a path to a JSON file or a dict already loaded.
        output_html_path: Path where the extracted HTML should be saved.

    Returns:
        Path to the saved HTML file.

    Raises:
        ValueError: If no ```html ... ``` block is found.
        KeyError: If expected keys ('vlm_output' / 'text') are missing.
    """
    # Load JSON if a path/string is provided
    if isinstance(json_source, (str, Path)):
        json_path = Path(json_source)
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_source

    # Get the text field
    text = data["vlm_output"]["text"]

    # Regex to capture content between ```html and ```
    match = re.search(
        r"```html\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not match:
        raise ValueError("No ```html ... ``` block found in vlm_output['text'].")

    # This keeps all internal newlines and spaces exactly as generated
    html_content = match.group(1)

    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
