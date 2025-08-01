#!/usr/bin/env bash
while true; do
  python -m orchestrator "Refactor utils.py for Python 3.12 compatibility"
  echo "Crash detected, restarting in 5s..." >&2
  sleep 5
done
