# pipeline_metrics.py
import time
from typing import List, Dict, Any


class PipelineMetrics:
    """
    Accumulates pipeline metrics between logging checkpoints.
    Tracks fetch, inference, and end-to-end batch processing times.
    """
    
    def __init__(self):
        # Fetch metrics
        self.fetch_times: List[float] = []  # milliseconds
        self.fetch_empty_count: int = 0
        self.fetch_wait_time: float = 0.0  # seconds
        
        # VLM inference metrics
        self.inference_times: List[float] = []  # milliseconds per batch
        self.inference_success_count: int = 0
        self.inference_failed_count: int = 0
        
        # End-to-end batch metrics
        self.batch_e2e_times: List[float] = []  # milliseconds (fetch + inference + enqueue)
        
        # Counters
        self.images_processed: int = 0
        self.batches_processed: int = 0

        # VLM output token/quality metrics (built worker-side; see core/wire.py)
        self.prompt_tokens: List[int] = []
        self.completion_tokens: List[int] = []
        self.mm_token_sums: Dict[str, int] = {}    # modality -> summed tokens
        self.mm_token_counts: Dict[str, int] = {}  # modality -> item count (avg = sums/counts)
        self.finish_reason_counts: Dict[str, int] = {}  # e.g. {"stop": N, "length": N}
        self.attempts_total: int = 0    # sum of retries consumed before each success
        self.attempts_count: int = 0    # number of successes attempts was recorded for

        # Timestamps
        self.window_start: float = time.time()
    
    def record_fetch(self, time_ms: float, is_empty: bool = False, wait_time_sec: float = 0.0):
        """
        Record a fetch operation.
        
        Args:
            time_ms: Time taken for fetch in milliseconds
            is_empty: Whether the fetch returned empty results
            wait_time_sec: Time spent waiting (if empty)
        """
        self.fetch_times.append(time_ms)
        if is_empty:
            self.fetch_empty_count += 1
            self.fetch_wait_time += wait_time_sec
    
    def record_inference(self, time_ms: float, batch_size: int, failed_count: int):
        """
        Record inference metrics.
        
        Args:
            time_ms: Time taken for inference in milliseconds
            batch_size: Number of images in the batch
            failed_count: Number of images that failed (returned None)
        """
        self.inference_times.append(time_ms)
        success_count = batch_size - failed_count
        self.inference_success_count += success_count
        self.inference_failed_count += failed_count
        self.images_processed += batch_size
        self.batches_processed += 1
    
    def record_tokens(self, full_vlm_obj: Dict[str, Any] = None):
        """
        Record per-item token/finish-reason/attempts stats from one
        successful result's full_vlm_object (built by the worker — see
        wire.stats_from_usage / wire.stats_from_request_output, whose outputs
        are key-identical across backends — and stamped in prediction._classify).
        No-op if unavailable (e.g. extraction failed for that item) so a
        missing field never breaks the running averages for the rest.
        """
        if not full_vlm_obj:
            return
        prompt_tokens = full_vlm_obj.get("prompt_tokens")
        if prompt_tokens is not None:
            self.prompt_tokens.append(prompt_tokens)
        completion_tokens = full_vlm_obj.get("completion_tokens")
        if completion_tokens is not None:
            self.completion_tokens.append(completion_tokens)
        mm_tokens = full_vlm_obj.get("mm_tokens")
        if mm_tokens:
            for modality, count in mm_tokens.items():
                self.mm_token_sums[modality] = self.mm_token_sums.get(modality, 0) + count
                self.mm_token_counts[modality] = self.mm_token_counts.get(modality, 0) + 1
        finish_reason = full_vlm_obj.get("finish_reason")
        if finish_reason is not None:
            self.finish_reason_counts[finish_reason] = self.finish_reason_counts.get(finish_reason, 0) + 1
        attempts = full_vlm_obj.get("attempts")
        if attempts is not None:
            self.attempts_total += attempts
            self.attempts_count += 1

    def record_batch_e2e(self, time_ms: float):
        """
        Record end-to-end batch processing time.
        
        Args:
            time_ms: Total time from fetch to enqueue in milliseconds
        """
        self.batch_e2e_times.append(time_ms)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Calculate and return summary statistics.
        
        Returns:
            Dictionary with average metrics for the current window
        """
        window_duration = time.time() - self.window_start
        total_images = self.inference_success_count + self.inference_failed_count
        total_inference_time = sum(self.inference_times)
        
        # Calculate rates and averages
        success_rate = (self.inference_success_count / total_images * 100) if total_images > 0 else 0.0
        failed_rate = (self.inference_failed_count / total_images * 100) if total_images > 0 else 0.0
        throughput = total_images / (total_inference_time / 1000) if total_inference_time > 0 else 0.0
        window_rate = self.images_processed / window_duration if window_duration > 0 else 0.0
        
        # Determine bottleneck
        avg_fetch = sum(self.fetch_times) / len(self.fetch_times) if self.fetch_times else 0.0
        avg_inference = total_inference_time / len(self.inference_times) if self.inference_times else 0.0
        
        return {
            "progress": {
                "images_processed_window": self.images_processed,
                "batches_processed_window": self.batches_processed,
                "window_duration_sec": round(window_duration, 2),
                "window_rate_imgs_per_sec": round(window_rate, 2)
            },
            "fetch_metrics": {
                "total_fetch_time_ms": round(sum(self.fetch_times), 2) if self.fetch_times else 0.0,
                "avg_fetch_time_ms": round(avg_fetch, 2),
                "max_fetch_time_ms": round(max(self.fetch_times), 2) if self.fetch_times else 0.0,
                "empty_fetches": self.fetch_empty_count,
                "total_wait_time_sec": round(self.fetch_wait_time, 2)
            },
            "vlm_metrics": {
                "total_inference_time_ms": round(total_inference_time, 2),
                "avg_inference_time_ms": round(avg_inference, 2),
                "avg_per_image_ms": round(total_inference_time / total_images, 2) if total_images > 0 else 0.0,
                "max_inference_time_ms": round(max(self.inference_times), 2) if self.inference_times else 0.0,
                "throughput_imgs_per_sec": round(throughput, 2),
                "success_count": self.inference_success_count,
                "failed_count": self.inference_failed_count,
                "success_rate_pct": round(success_rate, 2),
                "failed_rate_pct": round(failed_rate, 2)
            },
            "pipeline_metrics": {
                "avg_batch_e2e_ms": round(sum(self.batch_e2e_times) / len(self.batch_e2e_times), 2) if self.batch_e2e_times else 0.0,
                "bottleneck": "inference" if avg_inference > avg_fetch else "fetch"
            },
            "token_metrics": {
                "avg_prompt_tokens": round(sum(self.prompt_tokens) / len(self.prompt_tokens), 2) if self.prompt_tokens else 0.0,
                "avg_completion_tokens": round(sum(self.completion_tokens) / len(self.completion_tokens), 2) if self.completion_tokens else 0.0,
                "avg_mm_tokens_by_modality": {
                    modality: round(self.mm_token_sums[modality] / self.mm_token_counts[modality], 2)
                    for modality in self.mm_token_sums
                },
                "finish_reason_counts": dict(self.finish_reason_counts),
                "avg_retries_per_success": round(self.attempts_total / self.attempts_count, 2) if self.attempts_count else 0.0
            }
        }
    
    def reset(self):
        """Reset all metrics for the next window."""
        self.__init__()
