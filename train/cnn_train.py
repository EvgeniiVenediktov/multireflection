# %% [markdown]
# # Imports
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
from preprocess_images import data_from_folder
from tqdm import tqdm
from math import log

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchsummary import summary
import cv2 
import wandb
from config import LMDB_USE_COMPRESSION

import lmdb
import os
import msgpack
import io
import lz4.frame

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

torch.manual_seed(0)

# %%
def imname_to_target(name:str) -> tuple[float]:
    """Parses image names of format x{x_value}_y{y_value}.jpg"""
    name = name.split('.jpg')[0]
    x, y = name.split("_")
    x = float(x[1:])
    y = float(y[1:5])
    return x, y

def save_model(model:torch.nn.Module, fname="best_model.pth", path="./saved_models/real"):
    torch.save(model.state_dict(), os.path.join(path,fname))

def load_model(model:torch.nn.Module, fname="best_model.pth", path="./saved_models/real"):
    model.load_state_dict(torch.load(os.path.join(path,fname), weights_only=False))
    return model

# %% [markdown]
# # Config
# 

# %%
config = {
    "experiment_name": "001step_BS_actual-avid-sweep-4406_DarkOnly512_lmdb_400bs_0001lr_aug+",
    "batch_size": 400,
    "lr": 0.001,
    "lr_scheduler_loop": 7,
    "epochs": 96,
    "use_amp": False,

    "data_folder": "/mnt/h/black_512_0_001step_tensor.lmdb",
    # "data_folder": "/mnt/h/real_512_0_004step_tensor.lmdb",
    # "data_folder": "/mnt/e/color.lmdb",
    "dataset_type": "InMemoryLMDBImageDataset",
    "dataset_config_flatten": False,
    "dataset_train_keys_fname": "001_dark_train.txt",
    "dataset_val_keys_fname": "001_dark_val.txt",
    "dataset_offload_count": 0,

    "use_noise_transform": True,
    "noise_level": 0.1,
    "use_jitter_transform": True,
    "jitter_brightness": 0.4, 
    "jitter_contrast": 0.1, 
    "jitter_saturation": 0.1, 
    "jitter_hue": 0.2,

    "use_grayscale_transform": False,
    "use_clahegrad_transform": False,
    "clahe_clip_limit": 0.001,
    "clahe_gaussian_size": 15,
    "clahe_gaussian_sigma": 5,

    "use_high_pass_transform": False,
    "high_pass_transform_t": 0.35,

    "data_collection_step": 0.001,
    "starting_checkpoint_fname": None,
    "checkpoint_folder": "./saved_models",

    "gradient_layer_kernel_size": 15,
    "gradient_layer_sigma": 5,

    "use_weight_initialization": True,
    "init_red_filter": False
}

# %%

class InMemoryLMDBImageDataset(Dataset):
    def __init__(self, data_folder_path, transforms=None, keys_fname="keys.txt", flatten_data=True, turn_to_grayscale=True):
        self.keys = None

        # Data augmentation
        self.transforms = transforms

        # Read text keys from file
        with open(os.path.join(data_folder_path, keys_fname)) as f:
            self.keys = f.readlines()
            if self.keys[-1] == '':
                self.keys = self.keys[:-1]
        for i in range(len(self.keys)):
            self.keys[i] = self.keys[i].replace("\n", "")

        # Get labels from text keys
        self.labels = []
        for i, key in enumerate(self.keys):
            try:
                label = imname_to_target(key)

                # Convert label tuple to Tensor
                x, y = label
                x = (x + 2) / 5.7
                y = (y + 2) / 4
                label = (x, y)
                label = torch.tensor(label, dtype=torch.float32)
                self.labels.append(label)
            except Exception as e:
                print("i:", i)
                print("name:", key)
                raise e
            
        # Encode keys
        for i in range(len(self.keys)):
            self.keys[i] = self.keys[i].encode()

        # Load images
        self.env = lmdb.open(data_folder_path, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin()

        self.images = [None]*len(self.keys)
        self.loaded_indexes = set()
        self.flatten_data = flatten_data
        self.turn_to_grayscale = turn_to_grayscale

    def __len__(self):
        return len(self.keys)

    def get_index(self, key):
        for i, k in enumerate(self.keys):
            if k == key:
                return i
        
        return None
    
    def __getitem__(self, index):
        label = self.labels[index]

        if index in self.loaded_indexes:
            img = self.images[index]     
        else:
            key = self.keys[index]
            img_bytes = self.txn.get(key)
        
            if img_bytes is None:
                raise KeyError(f"Image {key} not found in LMDB!")


            img = torch.asarray(lmdb_bytes_to_torch_tensor(img_bytes), dtype=torch.float32)
            self.images[index] = img
            self.loaded_indexes.add(index)


        # Augmenation
        if self.transforms is not None:
            img = self.transforms(img)
        if self.flatten_data:
            img = img.flatten().float()
            self.debug_msg = f"image shape {img.shape}"
        elif isinstance(img, np.ndarray):
            img = torch.unsqueeze(torch.from_numpy(img), 0)

        return img, label


def lmdb_bytes_to_torch_tensor(img_bytes: bytes) -> torch.Tensor:
    if img_bytes is None:
        raise ValueError("img_bytes is None")

    buf = io.BytesIO(img_bytes)
    buf.seek(0)
    obj = torch.load(buf, map_location='cpu', weights_only=False)
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict) and 'tensor' in obj and isinstance(obj['tensor'], torch.Tensor):
        t = obj['tensor']
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
        elif isinstance(image, np.ndarray):
            image = torch.unsqueeze(torch.from_numpy(image), 0)
            # print(image.shape)

        # Convert label tuple to Tensor
        x, y = label
        x = (x + 2) / 5.7
        y = (y + 2) / 4
        label = (x, y)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

# %%
tarr = []

if config["use_jitter_transform"]:
    tarr.append(
        T.ColorJitter(
            config["jitter_brightness"],
            config["jitter_contrast"],
            # config["jitter_saturation"],
            # config["jitter_hue"]
        )
    )

if config["use_noise_transform"]:
    tarr.append(
        T.GaussianNoise(sigma=config["noise_level"]),
    )

varr = []

if config["use_grayscale_transform"]:
    tarr.append(T.Grayscale())
    varr.append(T.Grayscale())

train_transforms = T.Compose(tarr)
val_transforms = T.Compose(varr) if len(varr)>0 else None

# %%
match config["dataset_type"]:
    case "LMDBImageDataset":
        train_dataset = LMDBImageDataset(config["data_folder"], transforms=train_transforms, flatten_data=config["dataset_config_flatten"], keys_fname=config["dataset_train_keys_fname"])
        val_dataset = LMDBImageDataset(config["data_folder"], transforms=val_transforms, flatten_data=config["dataset_config_flatten"], keys_fname=config["dataset_val_keys_fname"])

    case "InMemoryImageDataset":
        train_dataset = InMemoryLMDBImageDataset(config["data_folder"], transforms=train_transforms, flatten_data=config["dataset_config_flatten"], keys_fname=config["dataset_train_keys_fname"])
        val_dataset = InMemoryLMDBImageDataset(config["data_folder"], transforms=val_transforms, flatten_data=config["dataset_config_flatten"], keys_fname=config["dataset_val_keys_fname"])
    case _ :
        raise("Wrong dataset type")
train_data_loader = DataLoader(train_dataset, 
                         batch_size=config["batch_size"], 
                         shuffle=True, 
                         num_workers=8, 
                         pin_memory=True, 
                         prefetch_factor=4, 
                         persistent_workers=True
                        )
val_data_loader = DataLoader(val_dataset,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             num_workers=4,
                             persistent_workers=True,
                             pin_memory=True
                            )

# %%
def size_after_conv(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1
conv_config = [
    {'out_channels':5, 'kernel_size':149, 'stride':15},
    {'out_channels':16, 'kernel_size':31, 'stride':10},
    {'out_channels':16, 'kernel_size':7, 'stride':2},
    # {'out_channels':16, 'kernel_size':6, 'stride':2},
    # {'out_channels':32, 'kernel_size':3, 'stride':2},
]
size_multiplier = 3

for l in conv_config:
    l['padding'] = l['kernel_size'] // 2

# %%
config['conv_config'] = conv_config
config['fc_head_size_multiplier'] = size_multiplier

# %%
class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50, ResNet-101, and ResNet-152"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, output_dim=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, output_dim)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Factory functions for different ResNet variants
def resnet18(output_dim=2):
    """ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], output_dim=output_dim)


def resnet34(output_dim=2):
    """ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], output_dim=output_dim)


def resnet50(output_dim=2):
    """ResNet-50 model"""
    return ResNet(Bottleneck, [3, 4, 6, 3], output_dim=output_dim)


def resnet101(output_dim=2):
    """ResNet-101 model"""
    return ResNet(Bottleneck, [3, 4, 23, 3], output_dim=output_dim)


def resnet152(output_dim=2):
    """ResNet-152 model"""
    return ResNet(Bottleneck, [3, 8, 36, 3], output_dim=output_dim)


# %%
class ConfigCNN(nn.Module):
    def __init__(self, output_size = 2, input_size=(1, 250, 250), size_multiplier=1):
        super(ConfigCNN, self).__init__()
        c, h, w = input_size
        layers = []
        prev_channels = c
        size = c * h * w
        for layer_config in conv_config:
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
            size = c * h * w * size_multiplier

        self.sec1 = nn.Sequential(
            *layers
        )

        self.sec2 = nn.Sequential(
            nn.Linear(size, size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(size // 2),
            nn.Linear(size // 2, output_size),
        )

    def forward(self, x):
        x = self.sec1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.sec2(x)

        return x


class SimpleFC(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleFC, self).__init__()
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 1024), # 262,144 -> 1024
            nn.BatchNorm1d(1024),
            self.relu,
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            self.relu,
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            self.relu,
            nn.Linear(32, out_features),
        )
    def forward(self, x):
        return self.layers.forward(x)
    
    

# model = SimpleFC(512*512, 2).to(DEVICE)
model = ConfigCNN(2, input_size=(1, 512, 512), size_multiplier=config['fc_head_size_multiplier']).to(DEVICE)
# model = resnet18(output_dim=2).to(DEVICE)

if config["use_weight_initialization"]:
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                          nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


summary(model, (1,512,512), config["batch_size"])
       
# %%
optimizer = optim.AdamW(model.parameters(), config["lr"], weight_decay=0.001)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config["lr_scheduler_loop"], eta_min=0.00001)
# scheduler = optim.lr_scheduler.ConstantLR(optimizer, 1, 0, )
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
# scaler = torch.cuda.amp.GradScaler("cuda", enabled=config["use_amp"])

# %%
wandb.login(key="a41d74c58ab2f0d2c2bbdb317450ab14a8ad9d4e")
wandb.init(
    project="multireflection",
    name=config["experiment_name"],
    config=config,
    resume="allow",
)
wandb.watch(model, log='all', log_freq=100)

# %%
from torch.amp import GradScaler, autocast

def train(model, train_loader, val_loader, optimizer: optim.Optimizer, criterion, scheduler: optim.lr_scheduler.CosineAnnealingWarmRestarts, best_loss=None):
    scaler = GradScaler(DEVICE)
    if best_loss is None:
        best_loss = 1000000000
    best_model = None
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.float16, enabled=False):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        last_lr = scheduler.get_last_lr()[0]
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast("cuda", dtype=torch.float16, enabled=False):
                    out = model(images)
                    loss = criterion(out, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        if avg_val_loss < best_loss:
            best_model = model
            best_loss = avg_val_loss
            save_model(model, fname=config["experiment_name"] + "_best_model.pth")

        print(f"Epoch {epoch + 1}/{config['epochs']}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # ✅ Log Training Loss
        log_train_loss = log(avg_train_loss)
        log_val_loss = log(avg_val_loss)
        avg_total_loss = avg_train_loss * 0.8 + avg_val_loss * 0.2
        log_total_loss = log(avg_total_loss)
        wandb.log({
            "Train Loss": avg_train_loss,
            "Val Loss": avg_val_loss,
            "LR": last_lr,
            "best_loss": best_loss,
            "log_train_loss": log_train_loss,
            "log_val_loss": log_val_loss,
            "avg_total_loss": avg_total_loss,
            "log_total_loss": log_total_loss,
            # "ds_train_loaded": len(train_dataset.loaded_indexes),
            # "ds_val_loaded": len(val_dataset.loaded_indexes),
        })

    print("Best loss:", best_loss)
    return model, best_model, best_loss


# %%
if config["starting_checkpoint_fname"] is not None:
    model = load_model(model, fname=config["starting_checkpoint_fname"], path=config["checkpoint_folder"])

# %%
best_loss = None
# best_loss = 0.0115

# %%
model, best_model, best_loss = train(model, train_data_loader, val_data_loader, optimizer, criterion, scheduler, best_loss)

# %% [markdown]
# # Bottom

# %%
# train_dataset.debug_msg

# %%
wandb.finish()


