#!/usr/bin/env bash
# Print REMAINING OpenRouter credit, or nothing if unreadable.
#
# WHY NOT /api/v1/key: on an uncapped key that endpoint reports a `usage` figure
# that is NOT the account's spend -- measured 2026-09-03, key usage $125.68 vs
# account total_usage $159.06, a $33 gap. A ceiling built on the key figure
# therefore lets the account overspend by whatever that gap happens to be, and
# the gap is not constant. /api/v1/credits gives total_credits and total_usage,
# and their difference is the only number that says when calls start failing.
KEY=$(grep -E "^OPEN_ROUTER_KEY=" /fsxvision_new/venkat.kesav/img_2_svg_pretraining/.env | cut -d= -f2- | tr -d '\r\n ')
curl -s --max-time 25 https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $KEY" \
| python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)['data']
    print(f\"{d['total_credits']-d['total_usage']:.2f}\")
except Exception:
    print('')" 2>/dev/null
