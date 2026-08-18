import argparse
import multiprocessing as mp
import os
import socket
import time
from types import SimpleNamespace
from vision_ingest.drivers.cli import cli
from vision_ingest.drivers.db_driver_sql import ingest_images
from vision_ingest.vllm_module import VLLMConfig, VLLMService
from project_specific import PromptAndValidation


def build_args():
    """
    Build configuration using SimpleNamespace instead of argparse.
    """
    return SimpleNamespace(

        # Database configuration
        db_path="/fsxvision/srihari.bandarupalli/Vision-Ingestion-Engine/examples/example2_12M_db/testing/db",  # REQUIRED – set explicitly
        logs_path="/fsxvision/srihari.bandarupalli/Vision-Ingestion-Engine/examples/example2_12M_db/testing/logs",
        jsonl_output_path="/fsxvision/srihari.bandarupalli/Vision-Ingestion-Engine/examples/example2_12M_db/testing/jsonl_outputs",

        # Image source configuration
        image_paths_source="/fsxvision/srihari.bandarupalli/Vision-Ingestion-Engine/examples/example2_12M_db/images.txt",
        ingest_batch_size = 32000, # SQLITE3 cant handle above this limit.
        # Processing configuration
        batch_size=4,
        local_json_threads=16,
        fsync_every_lines=250,

        # VLM configuration
        vlm_model_name="qwen_3_32b",
        vlm_gpus="0,1,2,3",
        vlm_config_path="/fsxvision/srihari.bandarupalli/Vision-Ingestion-Engine/src/vision_ingest/config/vllm_model.yaml",
        prompt_path="/fsxvision/srihari.bandarupalli/Vision-Ingestion-Engine/prompts/layout_v2.md",
    )

def model_init(args, logger, config_only=False):
    try:
        cfg_model = VLLMConfig(args.vlm_model_name, args.vlm_config_path)
        service = VLLMService(engine_args=cfg_model.engine_args,
                                  batch_size=args.batch_size,
                                  in_cache=True,
                                  lazy_loading=True,
                                  gpus=args.vlm_gpus
                                  )
        return cfg_model, service
    except Exception as e:
        if logger is None:
            print(f"Failed to initialize VLLMService: {e}")
        else:
            logger.error(f"Failed to initialize VLLMService: {e}")
        raise

def main_cli(args, vlm_config, vlm_service):

    run_id = time.strftime("%Y%m%d-%H%M%S")
    node_name = socket.gethostname()
    node_run_specific_log_dir = os.path.join(args.logs_path, node_name, run_id)
    args.logs_path = node_run_specific_log_dir

    ingest_images(args.db_path, args.image_paths_source, args.logs_path)

    print("----- Ingestion Done -----")
    prompt_validation_object = PromptAndValidation(args.prompt_path)
    cli(args, prompt_validation_object, vlm_config, vlm_service)

if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    # os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER' #'FLASH_ATTN' 
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp.set_start_method('spawn', force=True)
    
    args = build_args()
    vlm_config, vlm_service = model_init(args, logger=None)
    main_cli(args, vlm_config, vlm_service)
    vlm_service.shutdown()

