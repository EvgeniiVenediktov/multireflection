import os
import io
import lmdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import wandb


def imname_to_target(name: str) -> tuple[float, float]:
    name = name.split('.jpg')[0]
    x, y = name.split("_")
    x = float(x[1:])
    y = float(y[1:5])
    return x, y


def lmdb_bytes_to_torch_tensor(img_bytes: bytes) -> torch.Tensor:
    if img_bytes is None:
        raise ValueError("img_bytes is None")
    buf = io.BytesIO(img_bytes)
    buf.seek(0)
    obj = torch.load(buf, map_location='cpu')
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict) and 'tensor' in obj and isinstance(obj['tensor'], torch.Tensor):
        t = obj['tensor']
    else:
        raise ValueError("Unsupported LMDB object format: expected Tensor or dict with 'tensor'")
    return t.contiguous()


class LMDBImageDataset(Dataset):
    def __init__(self, lmdb_path, transforms=None, keys_fname="keys.txt", flatten_data=True):
        self.keys = None

        # Data augmentation
        self.transforms = transforms

        # Read text keys from file
        with open(os.path.join(lmdb_path, keys_fname)) as f:
            self.keys = f.readlines()
            if self.keys[-1] == '':
                self.keys = self.keys[:-1]
        for i in range(len(self.keys)):
            self.keys[i] = self.keys[i].replace("\n", "")

        # Get labels from text keys
        self.labels = []
        # self.labels = [imname_to_target(key) for key in self.keys]
        for i, key in enumerate(self.keys):
            try:
                self.labels.append(imname_to_target(key))
            except Exception as e:
                print("i:", i)
                print("name:", key)
                raise e

        # Encode keys
        for i in range(len(self.keys)):
            self.keys[i] = self.keys[i].encode()

        self.lmdb_path = lmdb_path
        self.flatten_data = flatten_data

    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin()

    def close(self):
        self.env.close()

    def __len__(self):
        return len(self.keys)

    def get_index(self, key):
        for i, k in enumerate(self.keys):
            if k == key:
                return i
        
        return None
    
    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            print("Opening lmdb txn")
            self.open_lmdb()
        key = self.keys[index]  # Get corresponding tuple
        label = self.labels[index]
        
        img_bytes = self.txn.get(key)
        
        if img_bytes is None:
            raise KeyError(f"Image {key} not found in LMDB!")


        image = torch.asarray(lmdb_bytes_to_torch_tensor(img_bytes), dtype=torch.float32)
        
        # Augmenation
        if self.transforms is not None:
            image = self.transforms(image)
            # print(f"image shaep after transforms: {image.shape}")
        if self.flatten_data:
            image = image.flatten().float()
            self.debug_msg = f"image shape {image.shape}"


        # Convert label tuple to Tensor
        x, y = label
        x = (x + 2) / 5.7
        y = (y + 2) / 4
        label = (x, y)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

class InMemoryLMDBImageDataset(Dataset):
    """LMDB-backed dataset that caches loaded tensors in memory.

    Keys are read from a text file inside the LMDB folder (defaults to `keys.txt`).
    Each key corresponds to an entry saved with `torch.save(tensor, buf)` in the LMDB.
    """
    def __init__(self, data_folder_path: str, transforms=None, keys_fname: str = "keys.txt", flatten_data: bool = True):
        self.transforms = transforms
        self.flatten_data = flatten_data
        self.data_folder_path = data_folder_path

        with open(os.path.join(data_folder_path, keys_fname), 'r') as f:
            keys = f.readlines()
        if len(keys) > 0 and keys[-1] == '':
            keys = keys[:-1]
        self.keys = [k.strip() for k in keys]

        self.labels = []
        for key in self.keys:
            self.labels.append(imname_to_target(key))

        # encoded keys used for lmdb lookup
        self.encoded_keys = [k.encode() for k in self.keys]

        self.env = lmdb.open(data_folder_path, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        if not self.env:
            raise ValueError(f"Cannot open LMDB folder at {data_folder_path}")
        self.txn = self.env.begin()
        if self.txn is None:
            raise ValueError(f"Cannot begin LMDB transaction at {data_folder_path}")

        self.images = [None] * len(self.keys)
        self.loaded_indexes = set()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int):
        label = self.labels[index]

        if index in self.loaded_indexes:
            img = self.images[index]
        else:
            key = self.encoded_keys[index]
            img_bytes = self.txn.get(key)
            if img_bytes is None:
                raise KeyError(f"Image {key} not found in LMDB!")
            img = torch.asarray(lmdb_bytes_to_torch_tensor(img_bytes), dtype=torch.float32)
            self.images[index] = img
            self.loaded_indexes.add(index)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.flatten_data:
            img = img.flatten().float()
        elif isinstance(img, torch.Tensor) and img.ndim == 2:
            img = img.unsqueeze(0)

        # normalize label to same scheme used in notebook
        x, y = label
        x = (x + 2) / 5.7
        y = (y + 2) / 4
        label_t = torch.tensor((x, y), dtype=torch.float32)

        return img, label_t


def size_after_conv(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1



class ConfigCNN(nn.Module):
    def __init__(self, conv_configuration, output_size=2, input_size=(1, 512, 512)):
        super(ConfigCNN, self).__init__()
        # conv_configuration: list of dicts with keys 'out_channels','kernel_size','stride' (padding optional)
        c, h, w = input_size
        layers = []
        prev_channels = c
        size = c * h * w
        # ensure padding present
        cfg = []
        for layer in conv_configuration:
            layer = dict(layer)
            if 'padding' not in layer:
                layer['padding'] = layer['kernel_size'] // 2
            cfg.append(layer)

        for layer_config in cfg:
            layers.append(
                nn.Conv2d(prev_channels,
                          layer_config['out_channels'],
                          layer_config['kernel_size'],
                          layer_config['stride'],
                          padding=layer_config['padding']
                          )
            )
            prev_channels = layer_config['out_channels']
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(layer_config['out_channels']))

            h = size_after_conv(h, layer_config['kernel_size'], layer_config['stride'], layer_config['padding'])
            w = h
            c = layer_config['out_channels']
            size = c * h * w

        self.sec1 = nn.Sequential(
            *layers
        )

        self.sec2 = nn.Sequential(
            nn.Linear(size, size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(size // 2),
            nn.Linear(size//2, output_size),
        )

    def forward(self, x):
        x = self.sec1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.sec2(x)

        return x


def make_config_cnn(conv_configuration, output_dim=2, input_size=(1, 512, 512)):
    return ConfigCNN(conv_configuration, output_size=output_dim, input_size=input_size)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running += loss.item()
    return running / len(loader)


def build_transforms(cfg):
    tarr = []
    # Data stored as tensors (C,H,W) or (H,W). Convert to proper dtype
    tarr.append(transforms.ConvertImageDtype(torch.float))
    if cfg.get('use_noise_transform', False):
        sigma = cfg.get('noise_level', 0.1)
        tarr.append(lambda x: x + sigma * torch.randn_like(x))
    return transforms.Compose(tarr)


def run_training(config=None):
    # config is a wandb.config-like dict
    cfg = config if config is not None else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Data
    data_folder = cfg.get('data_folder')
    if data_folder is None:
        raise ValueError('data_folder must be set in config and point to lmdb folder containing keys.txt')

    transforms_obj = build_transforms(cfg)
    ds_train = LMDBImageDataset(data_folder, transforms=transforms_obj, keys_fname=cfg.get('dataset_train_keys_fname','keys.txt'), flatten_data=False)
    print(f"Training dataset size: {len(ds_train)}")
    ds_val = LMDBImageDataset(data_folder, transforms=transforms_obj, keys_fname=cfg.get('dataset_val_keys_fname','keys_val.txt'), flatten_data=False)
    print(f"Validation dataset size: {len(ds_val)}")

    train_loader = DataLoader(ds_train, 
                              batch_size=cfg.get('batch_size', 32), 
                              shuffle=True, 
                              num_workers=4, 
                              pin_memory=True, 
                              persistent_workers=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.get('batch_size', 32), shuffle=False, num_workers=2, pin_memory=True, 
                              persistent_workers=True)

    # parse conv_config from sweep config (can be provided as JSON string)

    cfg['l1_padding'] = cfg['l1_kernel_size'] // 2
    cfg['l2_padding'] = cfg['l2_kernel_size'] // 2
    cfg['l3_padding'] = cfg['l3_kernel_size'] // 2

    conv_cfg = [
        {
            'out_channels': cfg['l1_out_channels'],
            'kernel_size': cfg['l1_kernel_size'],
            'stride': cfg['l1_stride'],
            'padding': cfg['l1_padding'],
        },
        {
            'out_channels': cfg['l2_out_channels'],
            'kernel_size': cfg['l2_kernel_size'],
            'stride': cfg['l2_stride'],
            'padding': cfg['l2_padding'],
        },
        {
            'out_channels': cfg['l3_out_channels'],
            'kernel_size': cfg['l3_kernel_size'],
            'stride': cfg['l3_stride'],
            'padding': cfg['l3_padding'],
        },
    ]

    model = make_config_cnn(conv_cfg, output_dim=2).to(device)
    if cfg.get('use_weight_initialization', True):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.get('lr', 1e-3), weight_decay=cfg.get('weight_decay', 0.0))
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cfg.get('lr_scheduler_loop', 7), eta_min=cfg.get('eta_min', 1e-5))

    best_val = float('inf')
    for epoch in range(cfg.get('epochs', 5)):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # step scheduler (mirror notebook behavior)
        if scheduler is not None:
            scheduler.step()

        # get lr for logging
        if scheduler is not None:
            try:
                last_lr = scheduler.get_last_lr()[0]
            except Exception:
                last_lr = optimizer.param_groups[0]['lr']
        else:
            last_lr = optimizer.param_groups[0]['lr']

        # log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': last_lr
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.save('best_model.pth')

    return best_val


def sweep_train():
    # This function is called by wandb.agent for each trial.
    with wandb.init() as run:
        cfg = wandb.config
        plain_cfg = dict(cfg)
        run_training(plain_cfg)


if __name__ == '__main__':
    # Sweep configuration: search over `conv_config` (JSON strings). Other training parameters mirror cnn_comparison defaults.
    sweep_config = {
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
            'noise_level': {'values': [0.1]},
            'use_jitter_transform': {'value': True},
            'jitter_brightness': {'value': 0.4},
            'jitter_contrast': {'value': 0.1},
            'jitter_saturation': {'value': 0.1},
            'jitter_hue': {'value': 0.2},

            'data_folder': {'value': './data/real_512_0_004step_tensor.lmdb'},
            'dataset_train_keys_fname': {'value': '004_dark_train.txt'},
            'dataset_val_keys_fname': {'value': '004_dark_val.txt'},
            'use_weight_initialization': {'value': True},
        }
    }
    torch.manual_seed(0)
    sweep_id = wandb.sweep(sweep_config, project='multireflection')
    wandb.agent(sweep_id, function=sweep_train, count=None)
