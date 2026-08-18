import argparse
import multiprocessing as mp
import signal
import os
from vision_ingest.drivers.cli import cli
from vision_ingest.drivers.db_driver_sql import ingest_images
from vision_ingest.vllm_module import VLLMConfig, VLLMService
from project_specific import PromptAndValidation

# Global list to track active processes
active_processes = []

def model_init(args, logger=None):
    try:
        cfg_model = VLLMConfig(args.vlm_model_name, args.vlm_config_path)
        service = VLLMService(
            engine_args=cfg_model.engine_args,
            batch_size=args.batch_size,
            in_cache=True,
            lazy_loading=True,
            gpus=args.vlm_gpus,
        )
        return cfg_model, service
    except Exception as e:
        if logger is None:
            print(f"Failed to initialize VLLMService: {e}")
        else:
            logger.error(f"Failed to initialize VLLMService: {e}")
        raise

def signal_handler(signum, frame):
    """Forward interruption signals to child processes."""
    for proc in active_processes:
        if proc and proc.is_alive():
            os.kill(proc.pid, signum)
    exit(1)

def parse_args():
    """Parse command line arguments for the vision ingestion pipeline."""

    parser = argparse.ArgumentParser(description="Vision Ingestion Engine Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['ingest', 'pipeline', 'both'], default='both',
                       help='Mode: ingest (add images to DB), pipeline (process from DB), or both (ingest then process)')
    
    # Parallel execution
    parser.add_argument('--parallel', action='store_true',
                       help='Run ingest and pipeline in parallel processes (only applicable with --mode both)')
    
    # Database configuration
    parser.add_argument('--db-path', type=str, required=True, help='Path for database storage (images.db and seen_cache.db will be created here)')
    
    # Multi-stage pipeline support
    parser.add_argument('--next-stage-db-path', type=str, default=None, help='Path to next stage database for multi-stage pipelines (optional)')
    parser.add_argument('--fetch-state', type=int, choices=[0, 4], default=0, help='State to fetch from: 0 for first stage, 4 for subsequent stages')

    # Image source configuration
    parser.add_argument('--image-paths-source', type=str, required=False, help='Source for images: folder path to walk OR file path containing image paths (one per line). Required for ingest and both modes.')
    
    # Output configuration
    parser.add_argument('--logs-path', type=str, default='logs', help='Directory for storing logs')
    parser.add_argument('--jsonl-output-path', type=str, default='jsonl_outputs', help='Directory for storing JSONL output files')
    
    # Processing configuration
    parser.add_argument( '--batch-size', type=int, default=32, help='Batch size for fetching from DB and running vLLM inference')
    parser.add_argument( '--local-json-threads', type=int, default=16, help='Number of threads for reading images/metadata and writing local JSONs')
    parser.add_argument( '--fsync-every-lines', type=int, default=250, help='Number of lines to write before fsyncing to disk')
    
    # VLM configuration
    parser.add_argument('--vlm-model-name', type=str, default='qwen_3_32b', help='VLM model name to use')
    parser.add_argument('--vlm-gpus', type=str, default='0,1,2,3', help='GPU IDs to use (comma-separated, e.g., "0,1,2,3")')
    parser.add_argument('--vlm-config-path', type=str, default='config/vllm_model.yaml', help='Path to vLLM model configuration file')
    parser.add_argument('--prompt-path', type=str, default='prompts/v1.md', help='Path to prompt template file')
    
    return parser.parse_args()

def run_pipeline_with_args(args):
    """Initialize VLM, prompt validation and run pipeline - for multiprocessing."""
    print(f"\n{'='*60}")
    print("Starting PIPELINE process")
    print(f"{'='*60}\n")
    vlm_config, vlm_service = model_init(args)
    prompt_validation_object = PromptAndValidation(args.prompt_path)
    try:
        cli(args, prompt_validation_object, vlm_config, vlm_service)
    finally:
        vlm_service.shutdown()

def main(args):
    global active_processes
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    ingest_proc = None
    pipeline_proc = None

    # Start ingestion process if needed
    if args.mode in ['ingest', 'both']:
        if not args.image_paths_source:
            raise ValueError("--image-paths-source is required for 'ingest' or 'both' modes")
        
        ingest_proc = mp.Process(target=ingest_images, args=(args.db_path, args.image_paths_source, args.logs_path))
        ingest_proc.start()
        active_processes.append(ingest_proc)
        
        # If not parallel, wait for ingestion to complete
        if not args.parallel:
            ingest_proc.join()
            print("\n✓ Ingestion completed\n")
    
    # Start pipeline process if needed
    if args.mode in ['pipeline', 'both']:
        pipeline_proc = mp.Process(target=run_pipeline_with_args, args=(args,))
        pipeline_proc.start()
        active_processes.append(pipeline_proc)
    
    # Wait for any running processes to complete
    if ingest_proc and ingest_proc.is_alive():
        ingest_proc.join()
    if pipeline_proc:
        pipeline_proc.join()

if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    # os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER' #'FLASH_ATTN' 
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    main(args)
