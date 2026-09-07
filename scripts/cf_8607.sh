#!/bin/bash
exec /home/venkat.kesav/bin/cloudflared tunnel --url http://127.0.0.1:8607 > /tmp/cf_8607.log 2>&1
