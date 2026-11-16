"""Small CUDA correctness check; no model training or memory stress test."""
import json
import subprocess
import torch
import torchvision
subprocess.run(['nvidia-smi'], check=True)
assert torch.cuda.is_available(), 'CUDA unavailable'
p = torch.cuda.get_device_properties(0)
assert 'A2000' in p.name, p.name
x = torch.tensor([1., 2.], device='cuda')
assert (x * x).sum().item() == 5
print(json.dumps(dict(torch=torch.__version__, torchvision=torchvision.__version__,
                      cuda=torch.version.cuda, gpu=p.name, vram_bytes=p.total_memory)))
