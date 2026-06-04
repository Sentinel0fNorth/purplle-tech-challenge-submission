from __future__ import annotations

import argparse
import json
import time
import urllib.request

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:  # nosec - local/API endpoint
        return json.loads(resp.read().decode("utf-8"))


def render(api_url: str, store_id: str) -> Panel:
    metrics = fetch_json(f"{api_url.rstrip('/')}/stores/{store_id}/metrics")
    anomalies = fetch_json(f"{api_url.rstrip('/')}/stores/{store_id}/anomalies")
    table = Table(title=f"Live Store Intelligence · {store_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Unique visitors", str(metrics["unique_visitors"]))
    table.add_row("Conversion rate", f"{metrics['conversion_rate'] * 100:.2f}%")
    table.add_row("Latest queue depth", str(metrics["queue_depth"]))
    table.add_row("Abandonment rate", f"{metrics['abandonment_rate'] * 100:.2f}%")
    table.add_row("Average dwell time", f"{metrics['avg_dwell_seconds']:.2f}s")
    table.add_row("Anomalies", str(len(anomalies)))
    return Panel(table, title="Connected to API", subtitle="Ctrl+C to stop")


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal dashboard for live metrics.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--store-id", default="STORE_BLR_002")
    parser.add_argument("--refresh", type=float, default=2.0)
    args = parser.parse_args()
    console = Console()
    with Live(render(args.api_url, args.store_id), console=console, refresh_per_second=2) as live:
        while True:
            time.sleep(args.refresh)
            live.update(render(args.api_url, args.store_id))


if __name__ == "__main__":
    raise SystemExit(main())
