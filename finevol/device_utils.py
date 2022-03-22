import torch


def set_device(device_id: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{device_id}"
    else:
        print("WARNING: PyTorch not recognizing CUDA device -> forcing CPU...")
        return "cpu"
