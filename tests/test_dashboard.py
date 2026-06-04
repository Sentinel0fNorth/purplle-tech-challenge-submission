import json
from unittest.mock import patch, MagicMock
import pytest
from pipeline.dashboard import render, main

def test_dashboard_render():
    mock_metrics = {
        "unique_visitors": 10,
        "conversion_rate": 0.25,
        "queue_depth": 3,
        "abandonment_rate": 0.12,
        "avg_dwell_seconds": 45.5,
    }
    mock_anomalies = []
    
    with patch("urllib.request.urlopen") as mock_url_open:
        mock_resp = MagicMock()
        mock_resp.read.side_effect = [
            json.dumps(mock_metrics).encode("utf-8"),
            json.dumps(mock_anomalies).encode("utf-8")
        ]
        mock_url_open.return_value.__enter__.return_value = mock_resp
        
        panel = render("http://localhost:8000", "STORE_BLR_002")
        assert panel is not None
        assert "Connected to API" in panel.title

def test_dashboard_main():
    mock_metrics = {
        "unique_visitors": 10,
        "conversion_rate": 0.25,
        "queue_depth": 3,
        "abandonment_rate": 0.12,
        "avg_dwell_seconds": 45.5,
    }
    mock_anomalies = []
    
    with patch("urllib.request.urlopen") as mock_url_open, \
         patch("time.sleep", side_effect=KeyboardInterrupt) as mock_sleep, \
         patch("pipeline.dashboard.Live") as mock_live:
         
        mock_resp = MagicMock()
        mock_resp.read.side_effect = [
            json.dumps(mock_metrics).encode("utf-8"),
            json.dumps(mock_anomalies).encode("utf-8"),
            json.dumps(mock_metrics).encode("utf-8"),
            json.dumps(mock_anomalies).encode("utf-8")
        ]
        mock_url_open.return_value.__enter__.return_value = mock_resp
        
        with pytest.raises(KeyboardInterrupt):
            main()
