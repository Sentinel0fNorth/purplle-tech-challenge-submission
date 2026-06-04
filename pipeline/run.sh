#!/usr/bin/env bash
set -euo pipefail
python -m pipeline.detect --layout data/real_store_layout.json --manifest data/video_manifest.json --api-url "${API_URL:-http://localhost:8000}" --out data/generated_events.jsonl
