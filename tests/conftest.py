import sys
import os
import types
import pytest
import torch

# Adiciona pasta raiz ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ===============================
# Mock Picamera2 e Serial
# ===============================
sys.modules['picamera2'] = types.SimpleNamespace(
    Picamera2=lambda: types.SimpleNamespace(
        start=lambda: None,
        capture_file=lambda path: open(path, 'wb').close()
    )
)

sys.modules['serial'] = types.SimpleNamespace(
    Serial=lambda *a, **kwargs: types.SimpleNamespace(
        write=lambda x: None,
        readline=lambda: b"pm25: 10, co: 0.1, co2: 400\n"
    )
)

# ===============================
# Fixture para mockar modelo PyTorch
# ===============================
@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    from main import model

    class FakeModel:
        def __init__(self, *args, **kwargs): pass
        def eval(self): return self
        def to(self, device): return self
        def __call__(self, img, sensors):
            return torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2]])  # retorna softmax fake
    monkeypatch.setattr('main.model', FakeModel())
