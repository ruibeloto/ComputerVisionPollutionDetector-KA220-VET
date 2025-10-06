# tests/test_model.py
import torch
from model import PollutionModel

def test_model_forward_with_valid_input():
    model = PollutionModel(num_classes=5)
    img = torch.randn(1, 3, 224, 224)  # batch=1, imagem RGB 224x224
    sensors = torch.randn(1, 3)        # pm25, co, co2
    out = model(img, sensors)
    assert out.shape == (1, 5)  # 5 classes
    assert torch.isfinite(out).all()

def test_model_only_image_mode():
    model = PollutionModel(num_classes=5)
    img = torch.randn(1, 3, 224, 224)
    sensors = torch.zeros(1, 3)
    out = model(img, sensors)
    assert out.shape == (1, 5)
