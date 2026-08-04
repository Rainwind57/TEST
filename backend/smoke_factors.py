import sys
sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)
for path in ["/api/health", "/api/select/factors", "/api/factors/catalog", "/api/data/sectors"]:
    r = c.get(path)
    print(path, r.status_code, type(r.json()).__name__)
    if path in ("/api/select/factors", "/api/factors/catalog"):
        d = r.json()
        print("  count:", len(d))
        # 校验每条都含必需字段
        bad = [x for x in d if not all(k in x for k in ("key", "label", "group", "direction", "format"))]
        print("  missing-field items:", len(bad), bad[:3])
