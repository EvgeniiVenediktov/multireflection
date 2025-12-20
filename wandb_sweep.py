import os
import io
import argparse
import lmdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import wandb

_DATA_CACHE = {}


def imname_to_target(name: str) -> tuple[float, float]:
    name = name.split('.jpg')[0]
    x, y = name.split("_")
    return float(x[1:]), float(y[1:5])


def lmdb_bytes_to_torch_tensor(img_bytes: bytes) -> torch.Tensor:
    if img_bytes is None:
        raise ValueError("img_bytes is None")
    buf = io.BytesIO(img_bytes)
    obj = torch.load(buf, map_location='cpu', weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.contiguous()
    if isinstance(obj, dict) and 'tensor' in obj:
        return obj['tensor'].contiguous()
    raise ValueError("Unsupported LMDB format")


class InMemoryLMDBImageDataset(Dataset):
    def __init__(self, data_folder: str, keys_fname: str = "keys.txt", transform=None):
        self.transform = transform
        
        with open(os.path.join(data_folder, keys_fname), 'r') as f:
            self.keys = [k.strip() for k in f.readlines() if k.strip()]
        
        self.labels = [imname_to_target(k) for k in self.keys]
        self.images = self._load_all(data_folder)
        print(f"Loaded {len(self.images)} images from {keys_fname}")

    def _load_all(self, data_folder):
        images = []
        env = lmdb.open(data_folder, readonly=True, create=False, lock=False, readahead=True, meminit=False)
        with env.begin() as txn:
            for key in self.keys:
                img_bytes = txn.get(key.encode())
                if img_bytes is None:
                    raise KeyError(f"Image {key} not found")
                images.append(lmdb_bytes_to_torch_tensor(img_bytes).float())
        env.close()
        return images

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 2:
            img = img.unsqueeze(0)
        
        if self.transform:
            img = self.transform(img)
        
        x, y = self.labels[idx]
        label = torch.tensor([(x + 2) / 5.7, (y + 2) / 4], dtype=torch.float32)
        return img, label


def size_after_conv(size, kernel, stride, padding):
    return (size - kernel + 2 * padding) // stride + 1


class ConfigCNN(nn.Module):
    def __init__(self, conv_config, output_size=2, input_size=(1, 512, 512)):
        super().__init__()
        c, h, w = input_size
        layers = []
        
        for cfg in conv_config:
            padding = cfg.get('padding', cfg['kernel_size'] // 2)
            layers.extend([
                nn.Conv2d(c, cfg['out_channels'], cfg['kernel_size'], cfg['stride'], padding),
                nn.ReLU(),
                nn.BatchNorm2d(cfg['out_channels'])
            ])
            h = size_after_conv(h, cfg['kernel_size'], cfg['stride'], padding)
            w = h
            c = cfg['out_channels']
        
        flat_size = c * h * w
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(flat_size, flat_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(flat_size // 2),
            nn.Linear(flat_size // 2, output_size),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))


class GaussianNoise:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
    def __call__(self, x):
        return x + self.sigma * torch.randn_like(x)


def build_transform(cfg):
    t = [transforms.ConvertImageDtype(torch.float)]
    if cfg.get('use_noise_transform'):
        t.append(GaussianNoise(cfg.get('noise_level', 0.1)))
    if cfg.get('use_jitter_transform'):
        t.append(transforms.ColorJitter(
            brightness=cfg.get('jitter_brightness', 0.4),
            contrast=cfg.get('jitter_contrast', 0.1),
            saturation=cfg.get('jitter_saturation', 0.1),
            hue=cfg.get('jitter_hue', 0.2)
        ))
    return transforms.Compose(t)


def get_dataloaders(cfg):
    cache_key = (cfg['data_folder'], cfg['dataset_train_keys_fname'], cfg['dataset_val_keys_fname'])
    
    if cache_key not in _DATA_CACHE:
        transform = build_transform(cfg)
        train_ds = InMemoryLMDBImageDataset(cfg['data_folder'], cfg['dataset_train_keys_fname'], transform)
        val_ds = InMemoryLMDBImageDataset(cfg['data_folder'], cfg['dataset_val_keys_fname'], transform)
        _DATA_CACHE[cache_key] = (train_ds, val_ds)
    
    train_ds, val_ds = _DATA_CACHE[cache_key]
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, 
                              num_workers=cfg.get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg.get('num_workers', 4), pin_memory=True)
    return train_loader, val_loader


def build_model(cfg, device):
    conv_cfg = [
        {'out_channels': cfg['l1_out_channels'], 'kernel_size': cfg['l1_kernel_size'], 'stride': cfg['l1_stride']},
        {'out_channels': cfg['l2_out_channels'], 'kernel_size': cfg['l2_kernel_size'], 'stride': cfg['l2_stride']},
        {'out_channels': cfg['l3_out_channels'], 'kernel_size': cfg['l3_kernel_size'], 'stride': cfg['l3_stride']},
    ]
    model = ConfigCNN(conv_cfg).to(device)
    
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for x, y in loader:
        total += criterion(model(x.to(device)), y.to(device)).item()
    return total / len(loader)


def run_training(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(cfg, device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, cfg['lr_scheduler_loop'], eta_min=cfg.get('eta_min', 1e-5)
    )
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    for epoch in range(cfg['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': scheduler.get_last_lr()[0]})
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return best_val


def sweep_train():
    with wandb.init():
        run_training(dict(wandb.config))


SWEEP_CONFIG = {
    'method': 'grid',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'l1_out_channels': {'values': [1, 3, 5]},
        'l1_kernel_size': {'values': [9, 31, 75, 149]},
        'l1_stride': {'values': [5, 15, 25]},
        'l2_out_channels': {'values': [1, 8, 16]},
        'l2_kernel_size': {'values': [7, 15, 31]},
        'l2_stride': {'values': [2, 5, 10]},
        'l3_out_channels': {'values': [1, 16, 32]},
        'l3_kernel_size': {'values': [3, 5, 7]},
        'l3_stride': {'values': [2]},
        'lr': {'value': 0.001},
        'batch_size': {'value': 100},
        'weight_decay': {'value': 0.001},
        'lr_scheduler_loop': {'value': 7},
        'epochs': {'value': 3},
        'use_noise_transform': {'value': True},
        'noise_level': {'value': 0.1},
        'use_jitter_transform': {'value': True},
        'jitter_brightness': {'value': 0.4},
        'jitter_contrast': {'value': 0.1},
        'jitter_saturation': {'value': 0.1},
        'jitter_hue': {'value': 0.2},
        'data_folder': {'value': './data/real_512_0_004step_tensor.lmdb'},
        'dataset_train_keys_fname': {'value': '004_dark_train.txt'},
        'dataset_val_keys_fname': {'value': '004_dark_val.txt'},
        'num_workers': {'value': 4},
    }
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, help='Resume existing sweep')
    parser.add_argument('--count', type=int, help='Max runs for this agent')
    parser.add_argument('--project', type=str, default='multireflection')
    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    sweep_id = args.sweep_id or wandb.sweep(SWEEP_CONFIG, project=args.project)
    print(f"Sweep ID: {sweep_id}")
    
    wandb.agent(sweep_id, function=sweep_train, count=args.count, project=args.project)