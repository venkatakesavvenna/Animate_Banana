# db_driver_sql.py
"""
Database management driver for image ingestion and maintenance.
"""

import argparse
import os
import time
import socket
from contextlib import contextmanager

from vision_ingest.db.db import DB

# Keep under SQLite's default parameter limit:
# Refer `Maximum Length Of An SQL Statement` at https://sqlite.org/limits.html for more details
# Reduce or increase the BATCH_SIZE based on the image_path size. Reduce it if u get TOO MANY VALUES error from sql
BATCH_SIZE = 100_000


@contextmanager
def db_operation(db_path: str, title: str, warn: bool = True):
    """Context manager for DB operations with header printing and error handling."""
    print(f"{title}...")
    print(f"Database path: {db_path}")
    if warn:
        print("⚠️  WARNING: Ensure no other processes are accessing the database!")
    print("-" * 80)
    
    db_file = os.path.join(db_path, "images.db")
    seen_cache_file = os.path.join(db_path, "seen_cache.db")
    db = DB(path=db_file, seen_path=seen_cache_file, verify=False)
    
    try:
        yield db
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        raise
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise
    finally:
        db.close()


def print_state_distribution(db: DB):
    health = db.get_db_health()
    print(f"\n📊 Database health:")
    print(f"   State 0 (pending): {health['state_distribution']['state_0']['count']:,}")
    print(f"   State 1 (in-progress): {health['state_distribution']['state_1']['count']:,}")
    print(f"   State 2 (done): {health['state_distribution']['state_2']['count']:,}")
    print(f"   State 3 (failed): {health['state_distribution']['state_3']['count']:,}")


def ingest_images(db_path: str, source: str, logs_path: str = "logs", batch_size: int = BATCH_SIZE, run_specific_log_path_in_args: bool = False):
    """Ingest images from source (folder or file) into database.
    
    Args:
        db_path: Path to database directory
        source: Path to folder to walk OR path to file containing image paths (one per line)
        logs_path: Path for storing logs
        batch_size: Number of paths per batch
    """
    # Initialize timestamped log directory
    if run_specific_log_path_in_args:
        node_run_specific_log_dir = logs_path
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        node_name = socket.gethostname()
        node_run_specific_log_dir = os.path.join(logs_path, f"{node_name}_ingest", run_id)
    
    print(f"Starting image ingestion from: {source}")
    print(f"Logs will be written to: {node_run_specific_log_dir}")
    print(f"Batch size: {batch_size}")
    
    with db_operation(db_path, "Initializing database", warn=False) as db:
        start_time = time.time()
        db.add_paths_from_source(source, batch_size=batch_size, logs_dir=node_run_specific_log_dir)
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"✅ Ingestion completed in {elapsed:.2f}s")
        health = db.get_db_health()
        print(f"📊 Final database health:")
        print(f"   Total images in main DB: {health['total_images_main']:,}")
        print(f"   Total images in seen cache: {health['total_images_seen']:,}")
        print(f"   Main DB size: {health['main_db_size_mb']:.2f} MB")
        print(f"   Seen cache size: {health['seen_cache_size_mb']:.2f} MB")


def verify_database(db_path: str):
    """Verify database counts and fix any drift."""
    with db_operation(db_path, "Starting database verification") as db:
        result = db.verify_and_fix_counts()
        
        if result['valid']:
            print("\n✅ Database verification passed - no issues detected")
        else:
            print("\n⚠️  Database had issues but they were fixed")


def reset_paths(db_path: str, reset_type: str):
    """Reset paths to state-0. reset_type: 'stuck' (state 1->0) or 'failed' (state 3->0)."""
    operations = {
        'stuck': ('Resetting stuck in-progress paths', lambda db: db.reset_stuck_in_progress()),
        'failed': ('Resetting failed paths', lambda db: db.reset_failed_paths())
    }
    
    title, operation = operations[reset_type]
    
    with db_operation(db_path, title) as db:
        start_time = time.time()
        count = operation(db)
        elapsed = time.time() - start_time
        
        print(f"\n✅ Reset {count:,} {reset_type} paths in {elapsed:.2f}s")
        
        if count > 0:
            print_state_distribution(db)


def full_maintenance(db_path: str, reset_failed: bool = False):
    """Run full maintenance: reset stuck paths + verify. Optionally reset failed paths."""
    print("=" * 80)
    print("FULL DATABASE MAINTENANCE")
    print("=" * 80)
    
    total_steps = 3 if reset_failed else 2
    
    # Step 1: Reset stuck paths
    print(f"\nStep 1/{total_steps}:")
    reset_paths(db_path, 'stuck')
    
    # Step 2 (optional): Reset failed paths
    if reset_failed:
        print(f"\n{'=' * 80}")
        print(f"\nStep 2/{total_steps}:")
        reset_paths(db_path, 'failed')
    
    # Final step: Verify and fix counts
    print(f"\n{'=' * 80}")
    print(f"\nStep {total_steps}/{total_steps}:")
    verify_database(db_path)
    
    print("\n" + "=" * 80)
    print("✅ Full maintenance completed")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Database management driver for PatramEDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest images from a folder
  python -m drivers.db_driver --db-path outputs/project/db --mode ingest --folder /path/to/images
  
  # Ingest images from a file containing paths (one per line)
  python -m drivers.db_driver --db-path outputs/project/db --mode ingest --folder /path/to/image_list.txt
  
  # Verify and fix database counts
  python -m drivers.db_driver --db-path outputs/project/db --mode verify
  
  # Reset stuck in-progress paths
  python -m drivers.db_driver --db-path outputs/project/db --mode reset-stuck
  
  # Reset failed paths
  python -m drivers.db_driver --db-path outputs/project/db --mode reset-failed
  
  # Run full maintenance (reset stuck + verify)
  python -m drivers.db_driver --db-path outputs/project/db --mode full-maintenance
  
  # Run full maintenance including failed path reset
  python -m drivers.db_driver --db-path outputs/project/db --mode full-maintenance --reset-failed
        """
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        required=True,
        help='Path to the database directory (images.db and seen_cache.db will be created here)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['ingest', 'verify', 'reset-stuck', 'reset-failed', 'full-maintenance'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        help='Source for image ingestion: folder path to walk OR file path containing image paths (one per line) [required for ingest mode]'
    )
    
    parser.add_argument(
        '--logs-path',
        type=str,
        default='logs',
        help='Path for storing logs (default: logs)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for ingestion (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--reset-failed',
        action='store_true',
        help='For full-maintenance mode: also reset failed paths (state 3 -> 0)'
    )
    
    args = parser.parse_args()
    
    # Create database directory if it doesn't exist
    os.makedirs(args.db_path, exist_ok=True)
    
    if args.mode == 'ingest':
        if not args.folder:
            parser.error("--folder is required for ingest mode")
        if not os.path.exists(args.folder):
            parser.error(f"Source does not exist: {args.folder}")
        ingest_images(args.db_path, args.folder, args.logs_path, args.batch_size)
        
    elif args.mode == 'verify':
        verify_database(args.db_path)
        
    elif args.mode == 'reset-stuck':
        reset_paths(args.db_path, 'stuck')
        
    elif args.mode == 'reset-failed':
        reset_paths(args.db_path, 'failed')
        
    elif args.mode == 'full-maintenance':
        full_maintenance(args.db_path, reset_failed=args.reset_failed)


if __name__ == "__main__":
    main()
