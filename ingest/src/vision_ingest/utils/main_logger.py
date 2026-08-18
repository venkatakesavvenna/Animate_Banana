# main_logger.py
import json
import time
import traceback
from typing import Dict, Any, Optional
from .pipeline_metrics import PipelineMetrics
from .utils import get_logger, _clean


class MainLogger:
    """
    Main logger class for pipeline operations.
    Provides structured JSON logging for startup, checkpoints, shutdown, and errors.
    """
    
    def __init__(self, log_dir: str, run_id: str):
        """
        Initialize the MainLogger.
        
        Args:
            log_dir: Directory where log files will be stored
            run_id: Unique run identifier (timestamp)
        """
        self.run_id = run_id
        self.logger = get_logger(log_dir, "main")
    
    def _log_json(self, level: str, data: Dict[str, Any]):
        """Log structured JSON data."""
        log_func = getattr(self.logger, level)
        log_func(json.dumps(_clean(data), ensure_ascii=False))
    
    def log_startup(
        self,
        config: Dict[str, Any],
        db_health,
        recovery_info: Optional[Dict[str, Any]] = None
    ):
        """
        Log pipeline startup event.
        
        Args:
            config: Configuration dictionary with paths and settings
            db: Database instance
            recovery_info: Optional recovery information (recovery_performed, last_shard_path, etc.)
        """
        self._log_json("info", {
            "component": "main",
            "event": "pipeline_started",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "config": config,
            "db_initial_state": db_health,
            **({"recovery_info": recovery_info} if recovery_info else {})
        })
    
    def log_batch(
        self,
        batch_type: str,
        batch_size: int,
        operation: str,
        duration_ms: float,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Log batch processing event (first/last/regular batch).
        
        Args:
            batch_type: Type of batch ('first', 'last', 'regular')
            batch_size: Number of images in the batch
            operation: Operation performed ('prompt_preparation', 'vlm_inference', 'parallel_processing')
            duration_ms: Duration of the operation in milliseconds
            additional_info: Optional additional information dictionary
        """
        self._log_json("info", {
            "component": "main",
            "event": "batch_processing",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "batch_type": batch_type,
            "batch_size": batch_size,
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            **({"additional_info": additional_info} if additional_info else {})
        })
    
    def log_checkpoint(
        self,
        metrics: PipelineMetrics,
        db_health,
        writer,
        cumulative_images: int,
        cumulative_failed: int,
        overall_start_time: float
    ):
        """
        Log processing checkpoint.
        
        Args:
            metrics: PipelineMetrics instance with accumulated metrics
            db: Database instance
            writer: JSONLWriter instance
            cumulative_images: Total images processed so far
            cumulative_failed: Total failed images so far
            overall_start_time: Pipeline start timestamp for overall rate calculation
        """
        # Get summary from metrics
        summary = metrics.get_summary()
        
        # Calculate overall rate and add to progress
        overall_duration = time.time() - overall_start_time
        overall_rate = cumulative_images / overall_duration if overall_duration > 0 else 0.0
        summary["progress"].update({
            "cumulative_images": cumulative_images,
            "overall_rate_imgs_per_sec": round(overall_rate, 2)
        })
        
        # Add cumulative failed count to VLM metrics
        summary["vlm_metrics"]["cumulative_failed"] = cumulative_failed
        
        # Get DB state and extract values
        db_state = db_health
        state_dist = db_state.get("state_distribution", {})
        total = db_state.get("total_images_main", 0)
        
        # Extract state counts with defaults
        state_counts = {
            f"state_{i}": state_dist.get(f"state_{i}", {}).get("count", 0)
            for i in range(4)  # states 0-3
        }
        
        # Calculate percentages
        state_2 = state_counts["state_2"]
        state_3 = state_counts["state_3"]
        completion_pct = (state_2 / total * 100) if total > 0 else 0.0
        failed_pct = (state_3 / total * 100) if total > 0 else 0.0
        
        # ETA calculation
        remaining = state_counts["state_0"]
        eta_hours = (remaining / overall_rate / 3600) if overall_rate > 0 else 0.0
        
        db_state_summary = {
            **{f"state_{i}_{['pending', 'processing', 'complete', 'failed'][i]}": state_counts[f"state_{i}"] 
               for i in range(4)},
            "total_images": total,
            "completion_pct": round(completion_pct, 2),
            "failed_pct": round(failed_pct, 2),
            "eta_hours": round(eta_hours, 2)
        }
        
        # Get writer state
        writer_state = {
            "current_shard_index": writer.shard_idx,
            "current_shard_size_mb": round(writer.current_size / (1024 * 1024), 2),
            "total_written_mb": round(
                (writer.shard_idx * writer.shard_bytes_limit + writer.current_size) / (1024 * 1024),
                2
            )
        }
        
        # Build and log checkpoint (single dict with unpacking)
        self._log_json("info", {
            "component": "main",
            "event": "processing_checkpoint",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            **summary,  # Unpacks progress, fetch_metrics, vlm_metrics, pipeline_metrics
            "db_state": db_state_summary,
            "writer_state": writer_state
        })
    
    def log_shutdown(
        self,
        reason: str,
        total_runtime: float,
        cumulative_images: int,
        cumulative_failed: int,
        db_health,
        writer
    ):
        """
        Log pipeline shutdown event.
        
        Args:
            reason: Shutdown reason (e.g., 'keyboard_interrupt', 'complete', 'error')
            total_runtime: Total pipeline runtime in seconds
            cumulative_images: Total images processed
            cumulative_failed: Total images that failed
            db: Database instance
            writer: JSONLWriter instance
        """
        # Calculate summary statistics
        avg_rate = cumulative_images / total_runtime if total_runtime > 0 else 0.0
        failed_rate = (cumulative_failed / cumulative_images * 100) if cumulative_images > 0 else 0.0
        
        self._log_json("info", {
            "component": "main",
            "event": "pipeline_stopped",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "reason": reason,
            "summary": {
                "total_images_processed": cumulative_images,
                "total_runtime_sec": round(total_runtime, 2),
                "avg_rate_imgs_per_sec": round(avg_rate, 2),
                "total_failed": cumulative_failed,
                "failed_rate_pct": round(failed_rate, 2),
                "final_shard_index": writer.shard_idx,
                "total_written_mb": round(
                    (writer.shard_idx * writer.shard_bytes_limit + writer.current_size) / (1024 * 1024),
                    2
                )
            },
            "db_final_state": db_health
        })
    
    def log_error(
        self,
        operation: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log error event with detailed context.
        
        Args:
            operation: Operation that failed (e.g., 'fetch_batch', 'mark_done', 'mark_failed')
            error: Exception object
            context: Optional context dictionary (e.g., batch_size, paths_count, etc.)
        """
        self._log_json("error", {
            "component": "main",
            "event": "operation_error",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "operation": operation,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            },
            **({"context": context} if context else {})
        })
