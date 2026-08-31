#!/usr/bin/env bash
# Bring up a 2-node Ray cluster inside the animatebanana-mn containers.
#   ROLE=head   bash ray_up.sh      # on 10.20.238.24
#   ROLE=worker bash ray_up.sh      # on 10.20.233.75
# Containers are --network host, so Ray advertises the real host IP and the
# worker can reach the head at a stable address.
set -u
C=animatebanana-mn
HEAD=${HEAD:-10.20.238.24}
PY=/environments/v5serve/bin
ROLE=${ROLE:?set ROLE=head|worker}

docker exec "$C" bash -lc "$PY/ray stop --force >/dev/null 2>&1; true"
if [ "$ROLE" = head ]; then
  docker exec "$C" bash -lc \
    "$PY/ray start --head --node-ip-address=$HEAD --port=6379 \
       --num-gpus=8 --disable-usage-stats 2>&1 | tail -3"
else
  docker exec "$C" bash -lc \
    "$PY/ray start --address=$HEAD:6379 --num-gpus=8 --disable-usage-stats 2>&1 | tail -3"
fi
