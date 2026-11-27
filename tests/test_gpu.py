import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config


def test_auto_device_configuration():
    config = load_config()
    assert config["gpu"]["device"] == "auto"
    assert config["training"]["batch_size"] == 1
    assert config["dataset"]["num_workers"] == 0
    assert config["gpu"]["mixed_precision"] is True
    assert config["gpu"]["pinned_memory"] is False
    assert config["training"]["cpu_num_threads"] == 2
    assert config["training"]["mode"] == "scratch"
    assert config["model"]["pretrained"] is False
