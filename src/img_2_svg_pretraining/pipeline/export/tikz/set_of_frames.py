import re
import os

def convert_to_multipage_pdf(tikz_code: str) -> str:
    """
    Converts animate-based TikZ code to a multi-page standalone format.
    """
    # 1. Update documentclass to force a new page for every frame
    # This specifically stops the horizontal rendering issue.
    tikz_code = re.sub(
        r'\\documentclass\[(.*?)\]\{standalone\}', 
        r'\\documentclass[\1,multi=tikzpicture]{standalone}', 
        tikz_code
    )
    # Fallback just in case there were no bracketed options originally
    if 'multi=tikzpicture' not in tikz_code:
        tikz_code = tikz_code.replace(
            r'\documentclass{standalone}', 
            r'\documentclass[multi=tikzpicture]{standalone}'
        )

    # 2. Remove the animate package
    tikz_code = re.sub(r'\\usepackage\{animate\}\n?', '', tikz_code)
    
    # 3. Remove the animateinline wrappers
    tikz_code = re.sub(r'\\begin\{animateinline\}.*\n', '', tikz_code)
    tikz_code = re.sub(r'\\end\{animateinline\}', '', tikz_code)
    
    # 4. Find the \multiframe declaration and extract Count and Start Index
    match = re.search(r'\\multiframe\{(\d+)\}\{iFrame=(\d+)\+1\}\{', tikz_code)
    if match:
        total_frames = int(match.group(1))
        start_index = int(match.group(2))
        
        # Calculate the correct final frame for the \foreach loop
        end_index = (start_index + total_frames) - 1
        
        # Replace the \multiframe line with the \foreach line
        foreach_replacement = f'\\foreach \\iFrame in {{{start_index},...,{end_index}}} {{'
        tikz_code = tikz_code[:match.start()] + foreach_replacement + tikz_code[match.end():]
        
    return tikz_code

if __name__ == "__main__":
    # ---------------------------------------------------------
    # HARDCODED CONFIGURATION
    # ---------------------------------------------------------
    input_filename = "/archive/medha.sen/temp_new/tex/elastic.tex"
    
    # Generate the output filename dynamically
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_export{ext}"
    
    # ---------------------------------------------------------
    # EXECUTION LOGIC
    # ---------------------------------------------------------
    try:
        # Read the generated LaTeX code
        with open(input_filename, 'r', encoding='utf-8') as f:
            original_code = f.read()
            
        # Apply the conversion
        converted_code = convert_to_multipage_pdf(original_code)
        
        # Write the resulting code to the new export file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(converted_code)
            
        print(f"Success! Converted '{input_filename}' into a multi-page sequence.")
        print(f"Saved to: {output_filename}")
        print(f"Next step: Run `pdflatex {output_filename}` then extract frames using pdftocairo.")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{input_filename}'.")
        print("Please ensure the file exists in the same directory as this script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")