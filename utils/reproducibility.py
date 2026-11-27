import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def capture_rng_state(train_loader=None, validation_loader=None):
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].astype(np.int64)),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch_cpu": torch.get_rng_state(),
        "train_loader": (
            train_loader.generator.get_state()
            if train_loader is not None and train_loader.generator is not None
            else None
        ),
        "validation_loader": (
            validation_loader.generator.get_state()
            if validation_loader is not None
            and validation_loader.generator is not None
            else None
        ),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, train_loader=None, validation_loader=None):
    if not isinstance(state, dict):
        raise ValueError("Checkpoint RNG state must be a mapping")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state[0],
        numpy_state[1].cpu().numpy().astype(np.uint32, copy=False),
        int(numpy_state[2]),
        int(numpy_state[3]),
        float(numpy_state[4]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "Checkpoint CUDA RNG state does not match the available "
                "GPU count."
            )
        torch.cuda.set_rng_state_all(cuda_states)
    if train_loader is not None and state.get("train_loader") is not None:
        if train_loader.generator is None:
            raise RuntimeError("Training DataLoader has no random generator")
        train_loader.generator.set_state(state["train_loader"])
    if (
        validation_loader is not None
        and state.get("validation_loader") is not None
    ):
        if validation_loader.generator is None:
            raise RuntimeError("Validation DataLoader has no random generator")
        validation_loader.generator.set_state(state["validation_loader"])
