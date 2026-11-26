from pathlib import Path
import yaml


def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent / "configs" / "p2pnet_scratch.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["model"].get("pretrained") is not False:
        raise RuntimeError("model.pretrained must be false")
    if config["training"].get("mode") != "scratch":
        raise RuntimeError("training.mode must be scratch")
    if config["training"].get("resume", False):
        raise RuntimeError("Checkpoint resume is forbidden for scratch training.")
    if config["training"].get("resume_checkpoint", ""):
        raise RuntimeError("Checkpoint loading is forbidden for training.")
    device = str(config["gpu"].get("device", "")).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("gpu.device must be 'auto', 'cpu', or 'cuda'")
    if config["gpu"].get("mixed_precision", False) and device == "cpu":
        raise RuntimeError("Mixed precision requires gpu.device=auto or cuda")
    if device == "cpu" and config["gpu"].get("pinned_memory", False):
        raise RuntimeError("Pinned memory must be disabled when gpu.device=cpu")
    if int(config["training"].get("cpu_num_threads", 0)) < 1:
        raise RuntimeError("training.cpu_num_threads must be a positive integer")
    if int(config["training"].get("cpu_interop_threads", 0)) < 1:
        raise RuntimeError("training.cpu_interop_threads must be a positive integer")
    if int(config["dataset"].get("num_workers", -1)) < 0:
        raise RuntimeError("dataset.num_workers cannot be negative")
    if int(config["training"].get("batch_size", 0)) < 1:
        raise RuntimeError("training.batch_size must be a positive integer")
    return config
