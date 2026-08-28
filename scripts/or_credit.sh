#!/usr/bin/env bash
# Remaining OpenRouter credit on the key in .env. The key carries a hard cap,
# so this is the budget, not an estimate.
set -euo pipefail
KEY=$(grep OPEN_ROUTER_KEY /fsxvision_new/venkat.kesav/img_2_svg_pretraining/.env | cut -d= -f2)
curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $KEY" | python3 -c "
import json,sys
d=json.load(sys.stdin)['data']
print(f\"used \${d['usage']:.4f} of \${d['limit']}  |  remaining \${d['limit_remaining']:.4f}\")"
