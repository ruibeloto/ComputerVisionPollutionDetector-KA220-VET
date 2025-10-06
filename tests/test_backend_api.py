# tests/test_backend_api.py
import pytest
from unittest.mock import MagicMock
import torch
from httpx import AsyncClient, ASGITransport

from main import app, model, CLASS_NAMES

model.forward = MagicMock(return_value=torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2]]))

@pytest.mark.asyncio
async def test_index_route_returns_html():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


@pytest.mark.asyncio
async def test_get_config_default_state():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/get_config")
    data = resp.json()
    assert "capture_active" in data
    assert "capture_mode" in data
    assert "capture_interval" in data


@pytest.mark.asyncio
async def test_latest_report_when_none():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/latest")
    data = resp.json()
    assert "error" in data or "message" in data


@pytest.mark.asyncio
async def test_capture_once_multimodal():
    transport = ASGITransport(app=app)
    import io
    from PIL import Image

    img_bytes = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
        data = {"mode": "multimodal", "pm25": 10, "co": 0.1, "co2": 400}
        resp = await ac.post("/capture_once", data=data, files=files)

    result = resp.json()
    assert "prediction_label" in result
    assert result["prediction_label"] in CLASS_NAMES
    assert "timestamp" in result
    assert "image_url" in result


@pytest.mark.asyncio
async def test_start_and_stop_capture():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # start_capture
        data = {"interval": 5, "mode": "multimodal"}
        resp = await ac.post("/start_capture", data=data)
        result = resp.json()
        assert result["status"] == "Captura iniciada"

        # stop_capture
        resp = await ac.post("/stop_capture")
        result = resp.json()
        assert result["status"] == "Captura parada"


@pytest.mark.asyncio
async def test_get_image_route():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/image.jpg")
        assert resp.status_code in (200, 422, 404)


@pytest.mark.asyncio
async def test_capture_once_image_only():
    transport = ASGITransport(app=app)
    import io
    from PIL import Image

    img_bytes = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(0, 255, 0))
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
        data = {"mode": "image_only", "pm25": 0, "co": 0, "co2": 0}
        resp = await ac.post("/capture_once", data=data, files=files)

    result = resp.json()
    assert "prediction_label" in result
    assert result["prediction_label"] in CLASS_NAMES
