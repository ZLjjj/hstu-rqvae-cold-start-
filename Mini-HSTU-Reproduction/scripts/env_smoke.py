import importlib

MODULES = ["torch", "accelerate", "torch_geometric", "hydra"]


def _import_lightning():
    try:
        module = importlib.import_module("lightning")
        return "lightning", getattr(module, "__version__", "unknown")
    except ModuleNotFoundError:
        module = importlib.import_module("pytorch_lightning")
        return "pytorch_lightning", getattr(module, "__version__", "unknown")


def main() -> None:
    loaded: dict[str, str] = {}
    name, version = _import_lightning()
    loaded[name] = version
    for name in MODULES:
        module = importlib.import_module(name)
        loaded[name] = getattr(module, "__version__", "unknown")

    import torch

    print("env_smoke: ok")
    for name in loaded:
        print(f"{name}: {loaded[name]}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
