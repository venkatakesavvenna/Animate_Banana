Main example demonstrating the vision_ingest package (installed from src/).

run_pipeline.sh: Automates Docker setup, environment creation, and pipeline execution.
main.py: Imports from vision_ingest package, includes signal handlers for clean shutdown (Ctrl+C).
project_specific.py: Custom PromptAndValidation class for project-specific logic.

The pipeline includes inner/outer cleanup to ensure all processes exit cleanly with no background threads.
This is the recommended template for large-scale runs—start, stop (Ctrl+C), and resume seamlessly from a single bash script.