.PHONY: test run simulate dashboard

test:
	pytest

run:
	uvicorn app.main:app --reload

simulate:
	python -m pipeline.detect --simulate --layout data/sample/store_layout.json --api-url http://localhost:8000 --out data/generated_events.jsonl

dashboard:
	python -m pipeline.dashboard --api-url http://localhost:8000 --store-id STORE_BLR_002
