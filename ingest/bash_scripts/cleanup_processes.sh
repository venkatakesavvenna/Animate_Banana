#!/bin/bash

# =============================================================================
# CLEANUP FUNCTION - Gracefully shutdown processes with escalating signals
# =============================================================================
# This function should be called inside Docker container to cleanup processes
# spawned by the pipeline script.
#
# IMPORTANT: This function will ALWAYS kill VLLM processes in addition to the
# specified process pattern, regardless of graceful shutdown success.
#
# Usage:
#   source bash_scripts/cleanup_processes.sh
#   PROCESS_PATTERN="python -m drivers"
#   trap 'cleanup_processes 30 "$PROCESS_PATTERN"' EXIT INT TERM
#
# Arguments:
#   $1 - GRACEFUL_TIMEOUT (optional, defaults to 30 seconds)
#   $2 - PROCESS_PATTERN (optional, defaults to "python -m drivers")
#
# Cleanup Strategy:
#   1. Send SIGINT to process groups (graceful / KeyboardInterrupt)
#   2. Wait for graceful shutdown (GRACEFUL_TIMEOUT seconds)
#   3. If still running, send SIGTERM (more forceful)
#   4. If still running, send SIGKILL (force kill)
#   5. ALWAYS kill VLLM processes at the end (regardless of above steps)
#
# Additionally:
#   - This implementation kills ALL processes matching the pattern
#   - It does NOT rely on head -1 or single PID
#   - It resolves full process groups to avoid orphaned children
#

cleanup_processes() {
    local GRACEFUL_TIMEOUT=${1:-30}
    local PROCESS_PATTERN=${2:-"python -m drivers"}

    echo "Cleanup: attempting graceful shutdown..."

    # =========================================================================
    # Collect ALL PIDs matching process pattern and ALL VLLM PIDs
    # =========================================================================
    mapfile -t MAIN_PIDS < <(pgrep -f "$PROCESS_PATTERN" || true)
    mapfile -t VLLM_PIDS < <(pgrep -f "VLLM" || true)

    if [ ${#MAIN_PIDS[@]} -eq 0 ] && [ ${#VLLM_PIDS[@]} -eq 0 ]; then
        echo "✓ No processes matching '$PROCESS_PATTERN' or 'VLLM' to clean up"
        return 0
    fi

    echo "Main process PIDs ($PROCESS_PATTERN): ${MAIN_PIDS[*]:-none}"
    echo "VLLM process PIDs: ${VLLM_PIDS[*]:-none}"

    # =========================================================================
    # Resolve ALL unique process groups
    # =========================================================================
    declare -A SEEN_PGID
    PGIDS=()

    for pid in "${MAIN_PIDS[@]}" "${VLLM_PIDS[@]}"; do
        [ -z "$pid" ] && continue
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ -n "$pgid" ] && [ -z "${SEEN_PGID[$pgid]}" ]; then
            SEEN_PGID[$pgid]=1
            PGIDS+=("$pgid")
        fi
    done

    echo "Process groups to signal: ${PGIDS[*]:-none}"

    # =========================================================================
    # STEP 1: Send SIGINT to ALL process groups
    # =========================================================================
    echo "Sending SIGINT to process groups (graceful shutdown)..."
    for pgid in "${PGIDS[@]}"; do
        kill -INT -"${pgid}" 2>/dev/null || true
    done

    echo "Waiting ${GRACEFUL_TIMEOUT}s for graceful shutdown..."
    for i in $(seq 1 "$GRACEFUL_TIMEOUT"); do
        alive=false
        for pid in "${MAIN_PIDS[@]}" "${VLLM_PIDS[@]}"; do
            [ -z "$pid" ] && continue
            if kill -0 "$pid" 2>/dev/null; then
                alive=true
                break
            fi
        done
        if [ "$alive" = false ]; then
            echo "✓ Graceful shutdown completed after ${i}s"
            break
        fi
        sleep 1
    done

    # =========================================================================
    # If anything is still alive → escalate
    # =========================================================================
    still_alive=false
    for pid in "${MAIN_PIDS[@]}" "${VLLM_PIDS[@]}"; do
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            still_alive=true
            break
        fi
    done

    # =========================================================================
    # STEP 2: SIGTERM (stronger but still allows cleanup)
    # =========================================================================
    if [ "$still_alive" = true ]; then
        echo "⚠️  Graceful timeout — sending SIGTERM to process groups..."
        for pgid in "${PGIDS[@]}"; do
            kill -TERM -"${pgid}" 2>/dev/null || true
        done
        sleep 5
    fi

    # =========================================================================
    # STEP 3: SIGKILL (force kill remaining)
    # =========================================================================
    echo "Force killing remaining processes..."

    # Kill based on pattern AND VLLM explicitly
    pkill -KILL -f "$PROCESS_PATTERN" 2>/dev/null || true
    pkill -KILL -f "VLLM" 2>/dev/null || true

    echo "✓ Cleanup completed"
}

# =============================================================================
# If script is sourced, export the function
# If script is executed directly, show usage
# =============================================================================
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    echo "This script should be sourced, not executed directly."
    echo ""
    echo "Usage:"
    echo "  source bash_scripts/cleanup_processes.sh"
    echo "  trap cleanup_processes EXIT INT TERM"
    echo ""
    echo "Or with custom timeout and pattern:"
    echo "  trap 'cleanup_processes 60 \"python -m myapp\"' EXIT INT TERM"
    exit 1
fi