# db_driver.py
"""
PostgreSQL database management driver for image ingestion and maintenance.

Modes:
  serve             Start PostgreSQL server (init cluster if needed) + heartbeat monitor
  stop              Stop PostgreSQL server (pg_ctl stop -m fast)
  serve-and-ingest  Start server + ingest images (heartbeat in daemon thread)
  ingest            Bulk-insert image paths into running server
  verify            Verify and fix state_counts drift
  reset-stuck       Reset state 1 -> 0
  reset-failed      Reset state 3 -> 0
  full-maintenance  reset-stuck + (optional) reset-failed + verify
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import psycopg # Ensure this is imported at the top of your file
import re      # Ensure this is imported at the top of your file

from vision_ingest.db.db import DB

BATCH_SIZE = 100_000

# Global flag for SIGTERM handling
_shutdown_requested = threading.Event()


def _signal_handler(signum, frame):
    """Handle SIGTERM by setting shutdown flag."""
    _shutdown_requested.set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pg_config(args) -> dict:
    """Build pg_config dict from parsed CLI args."""
    cfg = {
        "host": args.pg_host,
        "port": args.pg_port,
        "dbname": args.pg_dbname,
        "user": args.pg_user,
    }
    if args.pg_password:
        cfg["password"] = args.pg_password
    return cfg


def _run(cmd, check=True, **kwargs):
    """Run a subprocess command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def print_state_distribution(db: DB):
    health = db.get_db_health()
    dist = health["state_distribution"]
    print(f"\nDatabase health:")
    for state_key in sorted(dist.keys()):
        label = {
            "state_0": "pending",
            "state_1": "in-progress",
            "state_2": "done",
            "state_3": "failed",
            "state_4": "upstream-ready",
        }.get(state_key, state_key)
        print(f"   {state_key} ({label}): {dist[state_key]['count']:,}")

# ---------------------------------------------------------------------------
# Serve internals
# ---------------------------------------------------------------------------

def _serve_setup(data_dir: str, pg_config: dict, pg_password: str | None):
    """
    Initialize cluster, configure, start server, wait for ready, create DB.
    Runs once, then returns.
    """
    pg_user = pg_config["user"]
    pg_port = str(pg_config["port"])
    pg_dbname = pg_config["dbname"]
    pg_host = pg_config["host"]
    use_password = pg_password is not None
    auth_method = "md5" if use_password else "trust"

    # Step 1: initdb if needed
    pg_version_file = os.path.join(data_dir, "PG_VERSION")
    if not os.path.exists(pg_version_file):
        print("Initializing PostgreSQL cluster...")
        os.makedirs(data_dir, exist_ok=True)
        cmd = ["initdb", "-D", data_dir, f"--username={pg_user}"]
        if use_password:
            # Write password to temp file for --pwfile
            fd, pw_path = tempfile.mkstemp(prefix="pgpw_")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(pg_password)
                cmd.append(f"--pwfile={pw_path}")
                _run(cmd)
            finally:
                os.unlink(pw_path)
        else:
            cmd.append("--auth=trust")
            _run(cmd)
    else:
        print(f"Cluster already initialized at {data_dir}")

    # Step 2: Configure postgresql.conf
    conf_path = os.path.join(data_dir, "postgresql.conf")
    print(f"Configuring {conf_path}...")
    with open(conf_path, "r") as f:
        conf = f.read()

    # Set listen_addresses and port (replace if present, append if not)
    settings = {
        "listen_addresses": "'*'",
        "port": pg_port,
    }
    for key, value in settings.items():
        pattern = re.compile(rf"^\s*#?\s*{key}\s*=.*$", re.MULTILINE)
        replacement = f"{key} = {value}"
        if pattern.search(conf):
            conf = pattern.sub(replacement, conf)
        else:
            conf += f"\n{replacement}\n"
    with open(conf_path, "w") as f:
        f.write(conf)

    # Step 3: Configure pg_hba.conf
    hba_path = os.path.join(data_dir, "pg_hba.conf")
    print(f"Configuring {hba_path}...")
    with open(hba_path, "r") as f:
        hba = f.read()

    hba_lines = [
        f"host  all  {pg_user}  0.0.0.0/0  {auth_method}",
        f"host  all  {pg_user}  ::/0       {auth_method}",
    ]
    for line in hba_lines:
        if line not in hba:
            hba += f"\n{line}"
    with open(hba_path, "w") as f:
        f.write(hba)

    # Step 4: Start server
    log_path = os.path.join(data_dir, "postgres.log")
    print("Starting PostgreSQL server...")
    
    # UPDATED: Added "-o", "-k /tmp" to bypass the /var/run permission error
    _run(["pg_ctl", "start", "-D", data_dir, "-l", log_path, "-o", "-k /tmp"])

    # Step 5: Wait for server to be ready
    print("Waiting for server to accept connections...")
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            # UPDATED: Changed localhost to 127.0.0.1 to force TCP/IP and avoid socket confusion
            test_conn = psycopg.connect(
                host="127.0.0.1", port=int(pg_port), dbname="postgres", user=pg_user,
                **({"password": pg_password} if use_password else {}),
                connect_timeout=2,
            )
            test_conn.close()
            print("Server is ready.")
            break
        except psycopg.OperationalError:
            time.sleep(1)
    else:
        print("ERROR: Server did not become ready within 60 seconds.")
        print(f"Check log: {log_path}")
        sys.exit(1)

    # Step 6: Create database if it doesn't exist
    print(f"Ensuring database '{pg_dbname}' exists...")
    # UPDATED: Changed localhost to 127.0.0.1 here as well
    maint_conn = psycopg.connect(
        host="127.0.0.1", port=int(pg_port), dbname="postgres", user=pg_user,
        **({"password": pg_password} if use_password else {}),
        autocommit=True,
    )
    try:
        cur = maint_conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (pg_dbname,))
        if cur.fetchone() is None:
            cur.execute(f"CREATE DATABASE {pg_dbname}")
            print(f"Database '{pg_dbname}' created.")
        else:
            print(f"Database '{pg_dbname}' already exists.")
        cur.close()
    finally:
        maint_conn.close()

    # Step 7: Print connection summary
    print("\n" + "=" * 60)
    print("PostgreSQL server is running")
    print(f"  Host:     {pg_host}")
    print(f"  Port:     {pg_port}")
    print(f"  Database: {pg_dbname}")
    print(f"  User:     {pg_user}")
    print(f"  Auth:     {auth_method}")
    print(f"  Log:      {log_path}")
    print("=" * 60 + "\n")


def _heartbeat_loop(data_dir: str, pg_config: dict):
    """
    Blocking monitor loop. Every 30s: connect, print timestamp, DB size,
    active connections. Exits on Ctrl+C or SIGTERM.
    """
    import psycopg

    print("Heartbeat monitor started (every 30s). Press Ctrl+C to detach.")

    while not _shutdown_requested.is_set():
        try:
            conn = psycopg.connect(**pg_config, connect_timeout=5)
            try:
                cur = conn.cursor()
                cur.execute("SELECT pg_database_size(current_database()) / 1048576.0")
                size_mb = round(cur.fetchone()[0], 2)
                cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                active = cur.fetchone()[0]
                cur.close()
            finally:
                conn.close()
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] heartbeat OK | DB size: {size_mb} MB | active connections: {active}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] heartbeat FAILED: {e}")

        # Wait 30s, but check shutdown flag every 1s for responsiveness
        for _ in range(30):
            if _shutdown_requested.is_set():
                return  # serve()'s finally will call stop()
            time.sleep(1)

    print("\nHeartbeat monitor stopped. PostgreSQL server is still running.")
    print("Use 'db_driver stop' to stop the server.")


# ---------------------------------------------------------------------------
# Mode functions
# ---------------------------------------------------------------------------

def serve(data_dir: str, pg_config: dict, pg_password: str | None):
    _serve_setup(data_dir, pg_config, pg_password)
    try:
        _heartbeat_loop(data_dir, pg_config)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop(data_dir)
        except Exception as e:
            print(f"ERROR: Failed to stop database: {e}")
            print(f"Please stop it manually: pg_ctl stop -D {data_dir} -m fast")


def stop(data_dir: str):
    """Stop PostgreSQL server."""
    print("Stopping PostgreSQL server...")
    _run(["pg_ctl", "stop", "-D", data_dir, "-m", "fast"])
    print("Server stopped.")


def serve_and_ingest(data_dir: str, pg_config: dict, pg_password: str | None,
                     source: str, logs_path: str, batch_size: int,
                     use_copy: bool, initial_insert_state: int, run_specific_log_path_in_args: bool = False):
    """Start server, run heartbeat in daemon thread, ingest in main thread."""
    _serve_setup(data_dir, pg_config, pg_password)

    # Start heartbeat in daemon thread
    t = threading.Thread(target=_heartbeat_loop, args=(data_dir, pg_config), daemon=True)
    t.start()

    # Run ingest in main thread
    ingest_images(pg_config, source, logs_path, batch_size, use_copy, initial_insert_state, run_specific_log_path_in_args)

    print("\nIngestion complete. Server is still running.")
    print("Use 'db_driver stop' to stop the server.")


def ingest_images(pg_config: dict, source: str, logs_path: str = "logs",
                  batch_size: int = BATCH_SIZE, use_copy: bool = False,
                  initial_insert_state: int = 0, run_specific_log_path_in_args: bool = False):
    """Ingest images from source (folder or file) into database.
    
    Args:
        pg_config: PostgreSQL connection configuration dict
        source: Path to folder to walk OR path to file containing image paths (one per line)
        logs_path: Path for storing logs
        batch_size: Number of paths per batch
        use_copy: Use COPY FROM STDIN (fast, no dedup). Default: INSERT ON CONFLICT
        initial_insert_state: Initial state for inserted paths (default: 0)
        run_specific_log_path_in_args: If True, use logs_path as-is; if False, create timestamped subdirectory
    """
    if run_specific_log_path_in_args:
        node_run_specific_log_dir = logs_path
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        node_name = socket.gethostname()
        node_run_specific_log_dir = os.path.join(logs_path, f"{node_name}_ingest", run_id)

    print(f"Starting image ingestion from: {source}")
    print(f"Logs: {node_run_specific_log_dir}")
    print(f"Batch size: {batch_size:,}")
    print(f"Method: {'COPY' if use_copy else 'INSERT ON CONFLICT'}")
    print(f"Initial state: {initial_insert_state}")

    db = DB(pg_config=pg_config, fetch_state=initial_insert_state)
    try:
        start_time = time.time()
        db.add_paths_from_source(source, batch_size=batch_size,
                                 logs_dir=node_run_specific_log_dir,
                                 use_copy=use_copy)
        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print(f"Ingestion completed in {elapsed:.2f}s")
        health = db.get_db_health()
        print(f"Total images: {health['total_images_main']:,}")
        print(f"DB size: {health['main_db_size_mb']:.2f} MB")
        print_state_distribution(db)
    finally:
        db.close()


def verify_database(pg_config: dict):
    """Verify database counts and fix any drift."""
    print("Starting database verification...")
    db = DB(pg_config=pg_config)
    try:
        result = db.verify_and_fix_counts()
        if result["valid"]:
            print("\nVerification passed - no issues detected")
        else:
            print("\nDatabase had issues but they were fixed")
    finally:
        db.close()


def reset_paths(pg_config: dict, reset_type: str):
    """Reset paths to state-0. reset_type: 'stuck' (1->0) or 'failed' (3->0)."""
    operations = {
        "stuck": ("Resetting stuck in-progress paths (state 1 -> 0)", lambda db: db.reset_stuck_in_progress()),
        "failed": ("Resetting failed paths (state 3 -> 0)", lambda db: db.reset_failed_paths()),
    }
    title, operation = operations[reset_type]

    print(f"{title}...")
    db = DB(pg_config=pg_config)
    try:
        start_time = time.time()
        count = operation(db)
        elapsed = time.time() - start_time
        print(f"\nReset {count:,} {reset_type} paths in {elapsed:.2f}s")
        if count > 0:
            print_state_distribution(db)
    finally:
        db.close()


def full_maintenance(pg_config: dict, reset_failed_flag: bool = False):
    """Run full maintenance: reset stuck + verify. Optionally reset failed paths."""
    print("=" * 60)
    print("FULL DATABASE MAINTENANCE")
    print("=" * 60)

    total_steps = 3 if reset_failed_flag else 2

    # Step 1: Reset stuck paths
    print(f"\nStep 1/{total_steps}:")
    reset_paths(pg_config, "stuck")

    # Step 2 (optional): Reset failed paths
    if reset_failed_flag:
        print(f"\n{'=' * 60}")
        print(f"\nStep 2/{total_steps}:")
        reset_paths(pg_config, "failed")

    # Final step: Verify and fix counts
    print(f"\n{'=' * 60}")
    print(f"\nStep {total_steps}/{total_steps}:")
    verify_database(pg_config)

    print("\n" + "=" * 60)
    print("Full maintenance completed")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_connection_args(parser):
    """Add PostgreSQL connection arguments to a parser."""
    parser.add_argument("--pg-host", type=str, required=True,
                        help="Server IP/hostname (e.g. 10.0.0.1)")
    parser.add_argument("--pg-port", type=int, default=5432,
                        help="Server port (default: 5432)")
    parser.add_argument("--pg-dbname", type=str, required=True,
                        help="Database name (e.g. visiondb)")
    parser.add_argument("--pg-user", type=str, required=True,
                        help="Database user (e.g. pipeline)")
    parser.add_argument("--pg-password", type=str, default=None,
                        help="Database password (omit for trust auth)")


def _add_ingest_args(parser):
    """Add ingestion-specific arguments to a parser."""
    parser.add_argument("--source", type=str, required=True,
                        help="Folder to walk OR file of paths (one per line)")
    parser.add_argument("--logs-path", type=str, default="logs",
                        help="Path for storing logs (default: logs)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE:,})")
    parser.add_argument("--use-copy", action="store_true",
                        help="Use COPY FROM STDIN (fast, no dedup). Default: INSERT ON CONFLICT")
    parser.add_argument("--initial-insert-state", type=int, default=0,
                        help="Initial state for inserted paths (default: 0)")


def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL database driver for Vision Ingestion Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # Start PostgreSQL server
          python -m vision_ingest.drivers.db_driver serve \\
            --data-dir /data/pgdata --pg-host 10.0.0.1 --pg-dbname visiondb --pg-user pipeline

          # Stop server
          python -m vision_ingest.drivers.db_driver stop --data-dir /data/pgdata

          # Start server and ingest
          python -m vision_ingest.drivers.db_driver serve-and-ingest \\
            --data-dir /data/pgdata --pg-host 10.0.0.1 --pg-dbname visiondb --pg-user pipeline \\
            --source /path/to/images

          # Ingest from any node
          python -m vision_ingest.drivers.db_driver ingest \\
            --pg-host 10.0.0.1 --pg-dbname visiondb --pg-user pipeline \\
            --source /path/to/images

          # Verify database
          python -m vision_ingest.drivers.db_driver verify \\
            --pg-host 10.0.0.1 --pg-dbname visiondb --pg-user pipeline

          # Full maintenance
          python -m vision_ingest.drivers.db_driver full-maintenance \\
            --pg-host 10.0.0.1 --pg-dbname visiondb --pg-user pipeline --reset-failed
        """),
    )

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Operation mode")

    # --- serve ---
    p_serve = subparsers.add_parser("serve", help="Start PostgreSQL server + heartbeat monitor")
    p_serve.add_argument("--data-dir", type=str, required=True,
                         help="Path to PostgreSQL cluster data directory")
    _add_connection_args(p_serve)

    # --- stop ---
    p_stop = subparsers.add_parser("stop", help="Stop PostgreSQL server")
    p_stop.add_argument("--data-dir", type=str, required=True,
                        help="Path to PostgreSQL cluster data directory")

    # --- serve-and-ingest ---
    p_si = subparsers.add_parser("serve-and-ingest",
                                 help="Start server + ingest (heartbeat in daemon thread)")
    p_si.add_argument("--data-dir", type=str, required=True,
                      help="Path to PostgreSQL cluster data directory")
    _add_connection_args(p_si)
    _add_ingest_args(p_si)
    p_si.add_argument("--run-specific-log-path-in-args", action="store_true",
                      help="Use logs-path as-is instead of creating timestamped subdirectory")

    # --- ingest ---
    p_ingest = subparsers.add_parser("ingest", help="Bulk-insert image paths into running server")
    _add_connection_args(p_ingest)
    _add_ingest_args(p_ingest)
    p_ingest.add_argument("--run-specific-log-path-in-args", action="store_true",
                         help="Use logs-path as-is instead of creating timestamped subdirectory")

    # --- verify ---
    p_verify = subparsers.add_parser("verify", help="Verify and fix state_counts drift")
    _add_connection_args(p_verify)

    # --- reset-stuck ---
    p_rs = subparsers.add_parser("reset-stuck", help="Reset state 1 -> 0")
    _add_connection_args(p_rs)

    # --- reset-failed ---
    p_rf = subparsers.add_parser("reset-failed", help="Reset state 3 -> 0")
    _add_connection_args(p_rf)

    # --- full-maintenance ---
    p_fm = subparsers.add_parser("full-maintenance",
                                 help="reset-stuck + (optional) reset-failed + verify")
    _add_connection_args(p_fm)
    p_fm.add_argument("--reset-failed", action="store_true",
                      help="Also reset failed paths (state 3 -> 0) before verify")

    args = parser.parse_args()

    # Register SIGTERM handler
    signal.signal(signal.SIGTERM, _signal_handler)

    # Dispatch
    if args.mode == "serve":
        pg_config = _build_pg_config(args)
        serve(args.data_dir, pg_config, args.pg_password)

    elif args.mode == "stop":
        stop(args.data_dir)

    elif args.mode == "serve-and-ingest":
        pg_config = _build_pg_config(args)
        serve_and_ingest(args.data_dir, pg_config, args.pg_password,
                         args.source, args.logs_path, args.batch_size,
                         args.use_copy, args.initial_insert_state, args.run_specific_log_path_in_args)

    elif args.mode == "ingest":
        pg_config = _build_pg_config(args)
        ingest_images(pg_config, args.source, args.logs_path,
                      args.batch_size, args.use_copy, args.initial_insert_state, args.run_specific_log_path_in_args)

    elif args.mode == "verify":
        pg_config = _build_pg_config(args)
        verify_database(pg_config)

    elif args.mode == "reset-stuck":
        pg_config = _build_pg_config(args)
        reset_paths(pg_config, "stuck")

    elif args.mode == "reset-failed":
        pg_config = _build_pg_config(args)
        reset_paths(pg_config, "failed")

    elif args.mode == "full-maintenance":
        pg_config = _build_pg_config(args)
        full_maintenance(pg_config, reset_failed_flag=args.reset_failed)


if __name__ == "__main__":
    main()
