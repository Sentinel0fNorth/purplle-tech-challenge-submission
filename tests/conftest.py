import pytest
from fastapi.testclient import TestClient

@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.delenv("POS_CSV_PATH", raising=False)
    
    from app.db import init_db
    import app.db
    app.db.DB_PATH = str(db_path)
    
    from app.main import app

    init_db()
    with TestClient(app) as c:
        yield c


