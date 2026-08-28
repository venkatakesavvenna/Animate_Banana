#!/usr/bin/env bash
# All study suites. Run from the repo root.
#
#   bash scripts/run_study_tests.sh
#
# The LIVE suites need the server up on 8607:
#   bash scripts/run_study_server.sh
set -u
V=/fsxvision_new/venkat.kesav/environments/study/bin/python
# Chromium's shared libs are not installed system-wide (no root here); they
# live under the study venv, extracted from .debs.
export LD_LIBRARY_PATH=/fsxvision_new/venkat.kesav/environments/study/syslibs/usr/lib/x86_64-linux-gnu
export PYTHONPATH=src

# The LIVE suites assume a clean scratch database.
if [ "${RESTART:-1}" = "1" ]; then FRESH=1 bash scripts/run_study_server.sh >/dev/null 2>&1; sleep 2; fi

fail=0
for t in tests/test_study_bundle.py tests/test_study_scheduler.py \
         tests/test_study_app.py tests/test_study_ui.py; do
  printf '%-32s ' "$(basename "$t")"
  out=$($V "$t" 2>&1); rc=$?
  echo "$out" | tail -1
  [ $rc -ne 0 ] && { fail=1; echo "$out" | grep '^  FAIL' | sed 's/^/    /'; }
done
exit $fail
