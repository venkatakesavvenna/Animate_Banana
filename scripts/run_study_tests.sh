#!/usr/bin/env bash
# All study suites. Run from the repo root.
#
#   bash scripts/run_study_tests.sh
#
# The two offline suites need nothing. The two LIVE suites need one server per
# cohort, and they consume judgments (a full session per participant), so they
# must NEVER point at a production database. By default this starts two fresh
# scratch servers and stops them afterwards:
#
#   8612  main-v1      + study_main.yaml       data/study_runs/test_8612.db
#   8613  selective-v1 + study_selective.yaml  data/study_runs/test_8613.db
#
# Knobs:
#   RESTART=0     reuse whatever is already on MAIN_PORT / SEL_PORT (no wipe)
#   KEEP=1        leave the scratch servers running for inspection
#   MAIN_PORT / SEL_PORT   choose other ports (8612+ are free for this)
#   ONLY=ui       run a single suite: bundle | scheduler | app | ui
set -u
V=/fsxvision_new/venkat.kesav/environments/study/bin/python
# Chromium's shared libs are not installed system-wide (no root here); they
# live under the study venv, extracted from .debs.
export LD_LIBRARY_PATH=/fsxvision_new/venkat.kesav/environments/study/syslibs/usr/lib/x86_64-linux-gnu
export PYTHONPATH=src

MAIN_PORT=${MAIN_PORT:-8612}
SEL_PORT=${SEL_PORT:-8613}
export STUDY_BASE="http://localhost:$MAIN_PORT"
export STUDY_BASE_SELECTIVE="http://localhost:$SEL_PORT"
export STUDY_ADMIN_TOKEN=${ADMIN_TOKEN:-devtoken}

ONLY=${ONLY:-}
need_live=1
case "$ONLY" in bundle|scheduler) need_live=0;; esac

stop_port() {
  local pid
  pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
}

if [ $need_live = 1 ]; then
  # The LIVE suites assume a clean scratch database: they assert on per-
  # participant targets and on exp1 coming first, and a database whose cells
  # have all retired serves "done" immediately -- correct behaviour, failing test.
  if [ "${RESTART:-1}" = "1" ]; then
    FRESH=1 PORT=$MAIN_PORT DB=data/study_runs/test_$MAIN_PORT.db \
      bash scripts/run_study_server.sh >/dev/null 2>&1
    FRESH=1 PORT=$SEL_PORT DB=data/study_runs/test_$SEL_PORT.db \
      BUNDLE=data/study_bundles/selective-v1 \
      CONFIG=src/img_2_svg_pretraining/pipeline/configs/study_selective.yaml \
      bash scripts/run_study_server.sh >/dev/null 2>&1
  fi
  # Wait for readiness rather than sleeping a fixed amount. A watchdog may be
  # restarting the same port concurrently, and a fixed sleep raced it -- the
  # LIVE suites then bailed with "server not reachable" and printed no summary,
  # which reads exactly like a hang.
  for port in $MAIN_PORT $SEL_PORT; do
    for _ in $(seq 1 40); do
      curl -s -o /dev/null -m 3 "http://localhost:$port/" && break
      sleep 1
    done
    curl -s -o /dev/null -m 3 "http://localhost:$port/" \
      || { echo "server on $port never came up" >&2; exit 2; }
  done
fi

suites="tests/test_study_bundle.py tests/test_study_scheduler.py tests/test_study_app.py tests/test_study_ui.py"
[ -n "$ONLY" ] && suites="tests/test_study_$ONLY.py"

fail=0
# Sequentially, on purpose: the two LIVE suites share the scratch servers, and
# running them at once has produced `database is locked` 500s from sqlite.
for t in $suites; do
  printf '%-32s ' "$(basename "$t")"
  out=$($V "$t" 2>&1); rc=$?
  echo "$out" | grep -E 'checks,|not reachable' | tail -1
  echo "$out" | grep '^  KNOWN' | sed 's/^/    /'
  [ $rc -ne 0 ] && { fail=1; echo "$out" | grep '^  FAIL' | sed 's/^/    /'; }
done

if [ $need_live = 1 ] && [ "${KEEP:-0}" != "1" ] && [ "${RESTART:-1}" = "1" ]; then
  stop_port "$MAIN_PORT"; stop_port "$SEL_PORT"
fi
exit $fail
