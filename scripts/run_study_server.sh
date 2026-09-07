#!/usr/bin/env bash
# Start (or restart) the study server on 8607 against the pilot bundle.
#
# Kills by listening port rather than by command-line pattern: `pkill -f
# study.app` also matches the shell running it.
set -u
PORT=${PORT:-8607}
# The main-study bundle (30 figures x AnimateBanana / Qwen 3.8 / talk). Override
# BUNDLE and CONFIG together for the selective cohort (selective-v1 + study_selective.yaml).
BUNDLE=${BUNDLE:-data/study_bundles/main-v2}
# NOT /tmp: it gets swept, and a swept study database is lost data.
DB=${DB:-data/study_runs/study_test.db}
LOG=${LOG:-data/study_runs/server_$PORT.log}
CONFIG=${CONFIG:-src/img_2_svg_pretraining/pipeline/configs/study_main.yaml}
[ -z "$CONFIG" ] && CONFIG=src/img_2_svg_pretraining/pipeline/configs/study_main.yaml
V=/fsxvision_new/venkat.kesav/environments/study/bin/python
mkdir -p data/study_runs

OLD=$(ss -ltnp 2>/dev/null | grep ":$PORT" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
[ -n "$OLD" ] && { kill "$OLD"; sleep 1; echo "stopped $OLD"; }

# FRESH=1 wipes the scratch DB. The LIVE suites need one: they assert on
# per-participant targets and on exp1 coming first, and a database whose cells
# have all retired serves "done" immediately -- correct behaviour, failing test.
if [ "${FRESH:-0}" = "1" ]; then rm -f "$DB" "$DB"-wal "$DB"-shm; echo "wiped $DB"; fi

PYTHONPATH=src setsid nohup "$V" -m img_2_svg_pretraining.study.app \
  --bundle "$BUNDLE" --config "$CONFIG" \
  --db "$DB" --port "$PORT" --admin-token "${ADMIN_TOKEN:-devtoken}" \
  > "$LOG" 2>&1 < /dev/null &
sleep 3
head -2 "$LOG"
echo "http://localhost:$PORT/"
