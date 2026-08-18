"""
1. First make sure you start a db server on some node
```bash
rm -rf /opt/dlami/nvme/srihari.bandarupalli/db_test
```

```bash
python -m vision_ingest.drivers.db_driver_postgres serve \
  --data-dir /opt/dlami/nvme/srihari.bandarupalli/db_test \
  --pg-host 10.20.213.80 \
  --pg-port 54326 \
  --pg-dbname phase_3 \
  --pg-user pipeline
```

 --> Get the IP address of the host in which you are running the serve command
 --> You will run the serve command on host node only once.
 --> Please keep the data-dir a local nvme path only, otherwise it will be very slow.

2. cd into `examples/vllm_testing` and run this `python main.py` in all the nodes you want to run(including host node)
    - make sure to use the same details (including host) you used in serve command in all the nodes.
3. Try to keep the logs_path in `/opt/dlami/nvme/<some dir>`
    - make sure you have write permission in that dir
    - please dont keep fsxvision path as logs path because this will slow down file system for both you and others as it involves os.fsync()
4.  Jsonl path can be on fsxvision as we write to it only at fixed checkpoints.
"""
# Standard imports
import os, time, socket
import multiprocessing as mp
from types import SimpleNamespace

# Dependency imports
from vision_ingest.vllm_module import VLLMConfig, VLLMService
from vision_ingest.drivers.db_driver_postgres import ingest_images
from vision_ingest.drivers.cli import cli

from chandra import PromptAndValidation

def build_args():
    return SimpleNamespace(
        # DB details -> same details as what you used to serve.
        pg_host = "10.20.213.80",
        pg_port = "54326",
        pg_dbname = "phase_3",
        pg_user = "pipeline",
        pg_password = None,

        #logs path -> preferably nvme so that it does not slowdown the file system.
        logs_path = "/opt/dlami/nvme/srihari.bandarupalli/chandra_testing_logs/v1_old",
        jsonl_output_path = "/code/dependencies/Patram_ingest_old/examples/chandra_testing/v1_old",

        # Insert into the database.
        # image_paths_source = "/fsxvision_new/hrithik.sagar/OCR_ANNOTATIONS/CODEBASES/OCR_ANNOTATIONS/vlm_tagging/all_datasets.txt",
        image_paths_source ="/code/dependencies/Patram_ingest_old/examples/chandra_testing/images_regular.txt",
        ingest_batch_size= 100000, # batch size for insert only.. It can be high.
        use_copy = False, # can be True if you want faster insert but only if you are sure if image_paths_source has no duplicates with the paths already inserted into db.
        run_ingest = True, 
        
        # Processing configuration
        batch_size = 128,
        local_json_threads = 32,
        fsync_every_lines = 250,
        fetch_state = 0,

        #VLM Configuration
        vlm_model_name="datalab-to/chandra-ocr-2",
        vlm_gpus="0,1,2,3,4,5,6,7",
        vlm_config_path="/fsxvision_new/aryanjain.intern/Vision-Ingestion-Engine/examples/chandra/vllm_model.yaml",
        prompt_path=None # Chandra prompt is hardcoded in chandra.py by default
    )

def model_init(args, logger, config_only = False):
    try:
        cfg_model = VLLMConfig(args.vlm_model_name, args.vlm_config_path)
        service = VLLMService(engine_args=cfg_model.engine_args,
                                  batch_size=args.batch_size,
                                  in_cache=True,
                                  lazy_loading=False,
                                  gpus=args.vlm_gpus
                                  )
        return cfg_model, service
    except Exception as e:
        if logger is None:
            print(f"Failed to initialize VLLMService: {e}")
        else:
            logger.error(f"Failed to initialize VLLMService: {e}")
        raise    

def main_cli(args, vlm_config, vlm_service, run_ingest=False):

    run_id = time.strftime("%Y%m%d-%H%M%S")
    node_name = socket.gethostname()
    node_run_specific_log_dir = os.path.join(args.logs_path, node_name, run_id)
    args.logs_path = node_run_specific_log_dir

    if run_ingest:
        db_path_or_config = {
            'host': args.pg_host,
            'port': args.pg_port,
            'dbname': args.pg_dbname,
            'user': args.pg_user,
            **({"password": args.pg_password} if args.pg_password else {}),
        }
        ingest_images(db_path_or_config, args.image_paths_source, args.logs_path,args.ingest_batch_size, run_specific_log_path_in_args=True)
        print("----- Ingestion Done -----")
    else:
        print("----- Ingestion Skipped -----")
    prompt_validation_object = PromptAndValidation(args.prompt_path)
    cli(args, prompt_validation_object, vlm_config, vlm_service, run_specific_log_path_in_args=True)
    
    # Print the final OCR summary
    prompt_validation_object.print_summary()

if __name__ == "__main__":
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp.set_start_method('spawn', force=True)
    
    args = build_args()
    vlm_config, vlm_service = model_init(args, logger=None)
    main_cli(args, vlm_config, vlm_service, args.run_ingest)
    vlm_service.shutdown()
