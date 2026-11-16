"""Manual synthetic memory test, never runs automatically. No saved weights/data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from shortcut_experiment import build_model, seed_everything
seed_everything(42)
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
assert 'A2000' in p.name
print(p, flush=True)
model = build_model(dict(model='vgg11_bn', pretrained=False, dataset='colored_mnist')).cuda()
print('Grad-CAM layer:', dict(model.named_modules())['features.18'], flush=True)
optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.0001)
scaler = torch.amp.GradScaler('cuda')
x = torch.randn(128, 3, 224, 224, device='cuda')
y = torch.arange(128, device='cuda') % 10
try:
    for phase in ['train_amp', 'validation_amp', 'evaluation_fp32']:
        torch.cuda.reset_peak_memory_stats()
        model.train(phase == 'train_amp')
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(phase == 'train_amp'):
                with torch.autocast('cuda', dtype=torch.float16, enabled=phase != 'evaluation_fp32'):
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                if phase == 'train_amp':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            del loss
        torch.cuda.synchronize()
        print(phase, 'peak allocated GiB', torch.cuda.max_memory_allocated()/2**30,
              'peak reserved GiB', torch.cuda.max_memory_reserved()/2**30, flush=True)
except torch.cuda.OutOfMemoryError:
    print(torch.cuda.memory_summary(), flush=True)
    raise
