"""
gpu_check.py - confirm a GPU node can actually run this project's training stack.

    srun -M gpu -p l40s --gres=gpu:1 -c 8 --mem=32G -t 00:05:00 \
        ./.venv/bin/python cluster/gpu_check.py

Reports the torch build, the architectures the wheel was compiled for, the device's own
compute capability, and a short conv throughput number.

The architecture line is the interesting one. Neither the PyPI CUDA 13 wheel nor the CRCD
module python/pytorch_251_311_cu124 ships native sm_89 kernels, yet both run correctly on
the L40S: CUDA guarantees binary compatibility forward across minor revisions within a
major compute capability, so sm_86 cubins execute on sm_89. cuDNN and cuBLAS ship their own
Ada kernels regardless. If a future wheel drops sm_8x entirely this check will catch it.
"""

import time

import torch
import torch.nn as nn


def main() -> None:
    print(f"  torch  : {torch.__version__} | cuda {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("  no CUDA device visible (expected on a login node)")
        return

    archs = torch.cuda.get_arch_list()
    major, minor = torch.cuda.get_device_capability(0)
    device_arch = f"sm_{major}{minor}"
    print(f"  device : {torch.cuda.get_device_name(0)} ({device_arch})")
    print(f"  archs  : {' '.join(archs)}")
    print(f"  {device_arch} native in wheel: {device_arch in archs}"
          f"{'' if device_arch in archs else '  (runs via forward compatibility)'}")

    x = torch.randn(4096, 4096, device="cuda")
    result = (x @ x).sum().item()
    print(f"  matmul : {'OK' if result == result else 'NaN'}")

    # A stand-in for the real model's first layers, at the resolution actually trained on.
    model = nn.Sequential(
        nn.Conv2d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2),
    ).cuda().to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    torch.backends.cudnn.benchmark = True

    batch = 64
    images = torch.randn(batch, 1, 512, 512, device="cuda").to(
        memory_format=torch.channels_last)
    labels = torch.randn(batch, 2, device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(10):
        step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    iterations = 30
    for _ in range(iterations):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  conv stack throughput: {batch * iterations / elapsed:.0f} img/s")


if __name__ == "__main__":
    main()
