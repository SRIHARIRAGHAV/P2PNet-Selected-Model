import ast
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch

import torch
from torch.nn import UninitializedParameter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from models import P2PNet


WEIGHT_LOADING_CALLS = {
    "load_checkpoint",
    "load_pretrained",
    "load_scratch_checkpoint",
    "load_state_dict",
    "load_state_dict_from_url",
    "load_weights",
    "torch.load",
}


def _call_name(function):
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        prefix = _call_name(function.value)
        return "{}.{}".format(prefix, function.attr) if prefix else function.attr
    return ""


def inspect_initialization_sources():
    paths = [
        ROOT / "train.py",
        ROOT / "models" / "backbone.py",
        ROOT / "models" / "p2pnet.py",
    ]
    source_text = {}
    loading_calls = []
    pretrained_requests = []
    for path in paths:
        if not path.is_file():
            return False, ["Missing initialization source: {}".format(path)]
        text = path.read_text(encoding="utf-8")
        source_text[path] = text
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in WEIGHT_LOADING_CALLS:
                    loading_calls.append(
                        "{}:{} {}".format(path.name, node.lineno, name)
                    )
                for keyword in node.keywords:
                    if keyword.arg == "pretrained" and isinstance(
                        keyword.value,
                        ast.Constant,
                    ) and keyword.value.value is True:
                        pretrained_requests.append(
                            "{}:{} pretrained=True".format(path.name, node.lineno)
                        )
                    if keyword.arg == "weights" and not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is None
                    ):
                        pretrained_requests.append(
                            "{}:{} non-empty weights argument".format(
                                path.name,
                                node.lineno,
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                for imported in node.names:
                    if imported.name in WEIGHT_LOADING_CALLS:
                        loading_calls.append(
                            "{}:{} imported {}".format(
                                path.name,
                                node.lineno,
                                imported.name,
                            )
                        )

    text = "\n".join(source_text.values()).lower()
    external_reference_patterns = (
        r"shtecha\.pth",
        r"original_repo",
        r"https?://",
    )
    external_references = [
        pattern
        for pattern in external_reference_patterns
        if re.search(pattern, text)
    ]
    uses_torchvision_model = "torchvision.models" in text
    problems = []
    if loading_calls:
        problems.append("checkpoint/weight loader references: " + ", ".join(loading_calls))
    if pretrained_requests:
        problems.append("pretrained requests: " + ", ".join(pretrained_requests))
    if external_references:
        problems.append("external references: " + ", ".join(external_references))
    if uses_torchvision_model:
        problems.append("torchvision model factory referenced")
    return not problems, problems


def parameter_statistics(parameters):
    total = 0
    total_sum = 0.0
    total_squared_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    tensor_count = 0
    trainable_count = 0
    for parameter in parameters:
        values = parameter.detach()
        if values.numel() == 0:
            continue
        count = values.numel()
        values64 = values.to(dtype=torch.float64)
        total += count
        tensor_count += 1
        trainable_count += count if parameter.requires_grad else 0
        total_sum += values64.sum().item()
        total_squared_sum += values64.square().sum().item()
        minimum = min(minimum, values64.min().item())
        maximum = max(maximum, values64.max().item())

    if total == 0:
        return {
            "parameters": 0,
            "trainable": 0,
            "tensors": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    mean = total_sum / total
    variance = max(0.0, total_squared_sum / total - mean * mean)
    return {
        "parameters": total,
        "trainable": trainable_count,
        "tensors": tensor_count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": minimum,
        "max": maximum,
    }


def _add_result(results, name, passed, details):
    results.append({
        "name": name,
        "passed": bool(passed),
        "details": str(details),
    })


def main():
    config = load_config(ROOT / "configs" / "p2pnet_scratch.yaml")
    model_config = config["model"]
    seed = int(config["project"]["random_seed"])
    results = []

    if model_config.get("pretrained") is not False:
        _add_result(
            results,
            "pretrained initialization disabled",
            False,
            "model.pretrained must be false",
        )
        return _print_report(results)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        with patch(
            "torch.load",
            side_effect=AssertionError("Checkpoint load attempted"),
        ) as torch_load, patch.object(
            torch.nn.Module,
            "load_state_dict",
            side_effect=AssertionError("State-dict load attempted"),
        ) as state_dict_load:
            model = P2PNet(
                row=int(model_config["row"]),
                line=int(model_config["line"]),
                num_classes=int(model_config["num_classes"]),
                pretrained=False,
            )
            _add_result(
                results,
                "fresh P2PNet instance",
                True,
                "constructed with pretrained=False",
            )
            no_load_calls = (
                not torch_load.called
                and not state_dict_load.called
            )
            _add_result(
                results,
                "no checkpoint loaded during initialization",
                no_load_calls,
                "torch.load calls={}, load_state_dict calls={}".format(
                    torch_load.call_count,
                    state_dict_load.call_count,
                ),
            )

            torch.manual_seed(seed + 1)
            comparison_model = P2PNet(
                row=int(model_config["row"]),
                line=int(model_config["line"]),
                num_classes=int(model_config["num_classes"]),
                pretrained=False,
            )
    except Exception as error:
        _add_result(results, "fresh P2PNet instance", False, error)
        return _print_report(results)

    try:
        P2PNet(pretrained=True)
    except RuntimeError:
        pretrained_guard = True
    else:
        pretrained_guard = False
    no_torchvision_weights = (
        "torchvision.models" not in
        (ROOT / "models" / "backbone.py").read_text(encoding="utf-8")
    )
    _add_result(
        results,
        "no pretrained backbone weights requested",
        pretrained_guard and no_torchvision_weights,
        "pretrained=True is rejected; backbone is locally constructed",
    )

    model_parameters = dict(model.named_parameters())
    comparison_parameters = dict(comparison_model.named_parameters())
    differing_tensors = sum(
        not torch.equal(parameter.detach(), comparison_parameters[name].detach())
        for name, parameter in model_parameters.items()
    )
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    initialized = all(
        not isinstance(parameter, UninitializedParameter)
        and parameter.numel() > 0
        and torch.isfinite(parameter).all().item()
        and parameter.detach().float().std(unbiased=False).item() > 0
        for parameter in trainable_parameters
    )
    _add_result(
        results,
        "trainable parameters initialized randomly",
        initialized and differing_tensors > 0,
        "{} trainable parameters across {} tensors; {} tensors differ "
        "between independent initializations".format(
            sum(parameter.numel() for parameter in trainable_parameters),
            len(trainable_parameters),
            differing_tensors,
        ),
    )
    del comparison_model

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print("Total parameters: {:,}".format(total_parameters))
    print("Trainable parameters: {:,}".format(
        sum(parameter.numel() for parameter in trainable_parameters)
    ))

    groups = {
        "backbone": model.backbone.parameters(),
        "feature layers (FPN)": model.fpn.parameters(),
        "classification head": model.classification.parameters(),
        "point regression head": model.regression.parameters(),
    }
    for group_name, parameters in groups.items():
        stats = parameter_statistics(parameters)
        print(
            "{}: parameters={:,}, trainable={:,}, mean={:.6g}, std={:.6g}, "
            "min={:.6g}, max={:.6g}".format(
                group_name,
                stats["parameters"],
                stats["trainable"],
                stats["mean"],
                stats["std"],
                stats["min"],
                stats["max"],
            )
        )

    parameter_finiteness = all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    )
    _add_result(
        results,
        "all model parameters finite",
        parameter_finiteness,
        "checked {} parameters".format(total_parameters),
    )

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_index = int(config["gpu"]["gpu_device_index"])
        if device_index >= torch.cuda.device_count():
            _add_result(
                results,
                "model on CUDA",
                False,
                "configured GPU index {} is unavailable".format(device_index),
            )
            device = torch.device("cpu")
        else:
            device = torch.device("cuda", device_index)
            torch.cuda.set_device(device_index)
            model.to(device)
            on_cuda = all(
                parameter.device.type == "cuda"
                for parameter in model.parameters()
            )
            _add_result(
                results,
                "model on CUDA",
                on_cuda,
                "selected device {}".format(device),
            )
    else:
        device = torch.device("cpu")
        model.to(device)
        _add_result(
            results,
            "model on CUDA",
            False,
            "CUDA is unavailable; forward test will run on CPU",
        )

    height, width = map(int, model_config["input_resolution"])
    try:
        dummy = torch.randn(1, 3, height, width, device=device)
        model.eval()
        with torch.inference_mode():
            outputs = model(dummy)
        valid_outputs = (
            outputs["pred_logits"].ndim == 3
            and outputs["pred_points"].ndim == 3
            and outputs["pred_logits"].shape[:2]
            == outputs["pred_points"].shape[:2]
            and outputs["pred_logits"].shape[0] == 1
            and outputs["pred_points"].shape[-1] == 2
            and torch.isfinite(outputs["pred_logits"]).all().item()
            and torch.isfinite(outputs["pred_points"]).all().item()
        )
        _add_result(
            results,
            "dummy-image forward pass",
            valid_outputs,
            "logits={}, points={}, device={}".format(
                tuple(outputs["pred_logits"].shape),
                tuple(outputs["pred_points"].shape),
                outputs["pred_points"].device,
            ),
        )
    except (RuntimeError, ValueError, KeyError) as error:
        _add_result(results, "dummy-image forward pass", False, error)

    source_check, source_problems = inspect_initialization_sources()
    _add_result(
        results,
        "no pretrained/external weight references in initialization code",
        source_check,
        "no external/pretrained loading references found"
        if source_check else "; ".join(source_problems),
    )
    return _print_report(results)


def _print_report(results):
    print("\nScratch initialization verification")
    for result in results:
        label = "PASS" if result["passed"] else "FAIL"
        print("[{}] {}: {}".format(label, result["name"], result["details"]))
    passed = sum(result["passed"] for result in results)
    failed = len(results) - passed
    print("\nResult: {}/{} checks passed; {} failed.".format(
        passed,
        len(results),
        failed,
    ))
    print("OVERALL: {}".format("PASS" if failed == 0 else "FAIL"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
