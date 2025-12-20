import os
import io
import lmdb

import torch
import torch.nn as nn
import torch.optim as optim

import wandb


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


def load_dataset_to_gpu(data_folder: str, keys_fname: str, device: torch.device):
    with open(os.path.join(data_folder, keys_fname), 'r') as f:
        keys = [k.strip() for k in f.readlines() if k.strip()]
    
    labels = torch.tensor([imname_to_target(k) for k in keys], dtype=torch.float32)
    labels[:, 0] = (labels[:, 0] + 2) / 5.7
    labels[:, 1] = (labels[:, 1] + 2) / 4
    
    images = []
    env = lmdb.open(data_folder, readonly=True, create=False, lock=False, readahead=True, meminit=False)
    with env.begin() as txn:
        for key in keys:
            img_bytes = txn.get(key.encode())
            if img_bytes is None:
                raise KeyError(f"Image {key} not found")
            img = lmdb_bytes_to_torch_tensor(img_bytes).float()
            if img.ndim == 2:
                img = img.unsqueeze(0)
            images.append(img)
    env.close()
    
    images = torch.stack(images).to(device)
    labels = labels.to(device)
    print(f"Loaded {len(keys)} images to GPU from {keys_fname}, shape: {images.shape}")
    return images, labels


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


def train_epoch(model, images, labels, batch_size, optimizer, criterion, noise_sigma):
    model.train()
    n = images.size(0)
    perm = torch.randperm(n, device=images.device)
    total_loss = 0.0
    n_batches = 0
    
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        x, y = images[idx], labels[idx]
        
        if noise_sigma > 0:
            x = x + noise_sigma * torch.randn_like(x)
        
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.inference_mode()
def validate(model, images, labels, batch_size, criterion):
    model.eval()
    n = images.size(0)
    total_loss = 0.0
    n_batches = 0
    
    for i in range(0, n, batch_size):
        x, y = images[i:i+batch_size], labels[i:i+batch_size]
        total_loss += criterion(model(x), y).item()
        n_batches += 1
    
    return total_loss / n_batches


def main():
    wandb.init()
    cfg = dict(wandb.config)
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_imgs, train_labels = load_dataset_to_gpu(cfg['data_folder'], cfg['dataset_train_keys_fname'], device)
    val_imgs, val_labels = load_dataset_to_gpu(cfg['data_folder'], cfg['dataset_val_keys_fname'], device)
    
    model = build_model(cfg, device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, cfg['lr_scheduler_loop'], eta_min=cfg.get('eta_min', 1e-5)
    )
    criterion = nn.MSELoss()
    
    noise_sigma = cfg.get('noise_level', 0.1) if cfg.get('use_noise_transform') else 0
    batch_size = cfg['batch_size']
    
    best_val = float('inf')
    for epoch in range(cfg['epochs']):
        train_loss = train_epoch(model, train_imgs, train_labels, batch_size, optimizer, criterion, noise_sigma)
        val_loss = validate(model, val_imgs, val_labels, batch_size, criterion)
        scheduler.step()
        
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': scheduler.get_last_lr()[0]})
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    wandb.finish()


if __name__ == '__main__':
    main()