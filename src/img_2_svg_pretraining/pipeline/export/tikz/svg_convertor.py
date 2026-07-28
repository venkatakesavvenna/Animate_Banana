import re
import os

def prepare_for_svg_export(tikz_code: str) -> str:
    """
    Prepares animate-based TikZ code for SVG extraction via dvisvgm.
    Leaves the animate package and multiframe logic entirely intact.
    """
    # 1. Inject dvisvgm into the documentclass options
    if 'dvisvgm' not in tikz_code:
        # Handles cases with existing options like [border=10pt]
        tikz_code = re.sub(
            r'\\documentclass\[(.*?)\]\{standalone\}', 
            r'\\documentclass[dvisvgm,\1]{standalone}', 
            tikz_code
        )
        # Fallback if no bracketed options existed originally
        tikz_code = tikz_code.replace(
            r'\documentclass{standalone}', 
            r'\documentclass[dvisvgm]{standalone}'
        )

    # 2. Add 'transparency group' to any scope with opacity (if missing)
    def inject_transparency_group(match):
        options = match.group(1)
        # Only inject if opacity is defined but transparency group is missing
        if 'opacity' in options and 'transparency group' not in options:
            return f'\\begin{{scope}}[{options}, transparency group]'
        # Return exactly as it was if no change is needed
        return match.group(0)

    # Find all instances of \begin{scope}[...] and process them through the function
    tikz_code = re.sub(r'\\begin\{scope\}\[(.*?)\]', inject_transparency_group, tikz_code)
    
    return tikz_code

if __name__ == "__main__":
    # ---------------------------------------------------------
    # HARDCODED CONFIGURATION
    # ---------------------------------------------------------
    input_filename = "/archive/medha.sen/temp_new/tex/slide_bbox_new.tex"
    
    # Generate the output filename dynamically (e.g., animation_export.tex)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_svg_export{ext}"
    
    # ---------------------------------------------------------
    # EXECUTION LOGIC
    # ---------------------------------------------------------
    try:
        # Read the generated LaTeX code
        with open(input_filename, 'r', encoding='utf-8') as f:
            original_code = f.read()
            
        # Apply the conversion
        converted_code = prepare_for_svg_export(original_code)
        
        # Write the resulting code to the new export file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(converted_code)
            
        print(f"Success! Prepared '{input_filename}' for SVG export.")
        print(f"Saved to: {output_filename}")
        print(f"Next step: Compile with `latex` or `lualatex --output-format=dvi` then run `dvisvgm`.")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{input_filename}'.")
        print("Please ensure the file exists in the specific directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")