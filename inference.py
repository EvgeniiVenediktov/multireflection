import torch
import torch.nn as nn
import numpy as np
from numpy.typing import ArrayLike
from config import X_TILT_START, X_TILT_STOP, Y_TILT_START, Y_TILT_STOP, OPTIMUM_IMAGE_PATH_LIST
from skimage.metrics import structural_similarity as ssim


def evaluate_position(current_image: ArrayLike, optimums: list[ArrayLike]) -> float:
    results = []
    for optimum in optimums:
        results.append(ssim(current_image, optimum))
    return round(max(results), 2)

class GradientMagnitude(nn.Module):
    def __init__(self, kernel_size=15, sigma=5):
        super().__init__()
        # Sobel filters
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

        self.register_buffer('weight_x', sobel_x)
        self.register_buffer('weight_y', sobel_y)

        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel(kernel_size, sigma))

    def _create_gaussian_kernel(self, kernel_size, sigma):
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x):
        # Apply Gaussian blur
        x_blurred = torch.nn.functional.conv2d(x, self.gaussian_kernel, padding=self.gaussian_kernel.shape[-1] // 2)

        # Apply Sobel filtering
        grad_x = torch.nn.functional.conv2d(x_blurred, self.weight_x, padding=1)
        grad_y = torch.nn.functional.conv2d(x_blurred, self.weight_y, padding=1)

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # Normalize to [0, 1] per image
        B = grad_mag.shape[0]
        grad_mag_flat = grad_mag.view(B, -1)
        min_vals = grad_mag_flat.min(dim=1)[0].view(B, 1, 1, 1)
        max_vals = grad_mag_flat.max(dim=1)[0].view(B, 1, 1, 1)
        grad_mag = (grad_mag - min_vals) / (max_vals - min_vals + 1e-6)

        return grad_mag
    
class GradientSimpleFC(nn.Module):
    def __init__(self, in_features, out_features):
        super(GradientSimpleFC, self).__init__()
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
            GradientMagnitude(),
            nn.Flatten(),
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
    

class WideConv(nn.Module):
    def __init__(self):
        super(WideConv, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 5, padding=2)
        
        # Instead of flattening, use global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Predicts 2 values
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)
        


class TiltPredictor:
    
    def load_model(self, model:torch.nn.Module, fname="best_model.pth", path="./saved_models/") -> nn.Module:
        model.load_state_dict(torch.load(path+fname, weights_only=False, map_location=self.DEVICE))
        return model
    
    def __init__(self, model_fname:str, model_type:str):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.DEVICE)

        match model_type:
            case "SimpleFC":
                self.model = SimpleFC(512*512, 2)
            case "WideConv":
                self.model = WideConv()
            case "GradientSimpleFC":
                self.model = GradientSimpleFC(512*512, 2)
            case _ :
                raise KeyError("Not supported model type")
        self.model_type = model_type
        self.model = self.load_model(self.model, fname=model_fname)
        self.model.eval()
        self.model.to(self.DEVICE)
    
    def predict(self, inputs:list[ArrayLike], scale_predictions=True) -> list[tuple[float]]:
        """
        Takes a list of (512, 512) images \n
        Returns list of [x_deg, y_deg]
        """
        tensors = []
        for img in inputs:
            img = torch.from_numpy(img).float()/255
            if self.model_type == "SimpleFC":
                img = img.flatten()
            tensors.append(img)
        x = torch.stack(tensors).to(self.DEVICE)

        predictions:torch.Tensor = self.model.forward(x)
        predictions = predictions.detach().cpu().numpy()
        if scale_predictions:
            predictions[:,0] = predictions[:,0] * (X_TILT_STOP - X_TILT_START) + X_TILT_START
            predictions[:,1] = predictions[:,1] * (Y_TILT_STOP - Y_TILT_START) + Y_TILT_START
        return predictions


if __name__=="__main__":
    import time
    model = TiltPredictor("fc_4layers_1024batch_500epochs_50cosinescheduler_best_model.pth", model_type="SimpleFC")
    x = np.zeros((125, 125))
    start = time.time()
    y = model.predict([x])
    print(y)
    print(f"Time elapsed: {time.time()-start:.2f}s")

