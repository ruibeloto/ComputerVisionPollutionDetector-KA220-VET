import builtins
import types
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.modules['picamera2'] = types.SimpleNamespace(Picamera2=lambda: types.SimpleNamespace(start=lambda: None, capture_file=lambda path: open(path,'wb').close()))
sys.modules['serial'] = types.SimpleNamespace(Serial=lambda *a, **k: types.SimpleNamespace(write=lambda x: None, readline=lambda: b"pm25: 10, co: 0.1, co2: 400\n"))

mock_rc = types.SimpleNamespace(
    read_sensors=lambda: {"pm25": 10, "co": 0.1, "co2": 400},
    capture_image=lambda: "fake.jpg",
    send_to_server=lambda pm25, co, co2, mode, img_path: {"ok": True},
    SLEEP_INTERVAL=0
)

sys.modules['raspberry_client'] = mock_rc

import raspberry_client as rc

def test_read_sensors_parses_values():
    import raspberry_client as rc
    data = rc.read_sensors()
    assert "pm25" in data
    assert isinstance(data["pm25"], (int, float)) 
