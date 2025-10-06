# tests/test_integration_e2e.py
import io
from PIL import Image
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_full_capture_flow():
    # 1. Verifica config inicial
    cfg = client.get("/get_config").json()
    assert cfg["capture_active"] is False

    # 2. Inicia captura
    client.post("/start_capture", data={"interval": 5, "mode": "multimodal"})

    # 3. Faz captura única
    img = Image.new("RGB", (224, 224), "blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    files = {"image": ("fake.jpg", buf, "image/jpeg")}
    data = {"mode": "multimodal", "pm25": "12", "co": "0.5", "co2": "420"}
    report = client.post("/capture_once", files=files, data=data).json()
    assert "prediction_label" in report

    # 4. Último relatório deve estar disponível
    latest = client.get("/latest").json()
    assert latest["prediction_label"] == report["prediction_label"]
