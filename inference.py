import torch
import torch.nn as nn
import numpy as np
from numpy.typing import ArrayLike

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
            case _ :
                raise KeyError("Not supported model type")
        self.model_type = model_type
        self.model = self.load_model(self.model, fname=model_fname)
        self.model.eval()
        self.model.to(self.DEVICE)
    
    def predict(self, inputs:list[ArrayLike]) -> list[tuple[float]]:
        """
        Takes a list of (125, 125) images \n
        Returns list of [x_deg, y_deg]
        """
        tensors = []
        for img in inputs:
            img = torch.from_numpy(img).float()
            if self.model_type == "SimpleFC":
                img = img.flatten()
            tensors.append(img)
        x = torch.stack(tensors).to(self.DEVICE)

        predictions:torch.Tensor = self.model.forward(x)
        predictions = predictions.detach().cpu().numpy()
        return predictions


if __name__=="__main__":
    import time
    model = TiltPredictor("fc_4layers_1024batch_500epochs_50cosinescheduler_best_model.pth", model_type="SimpleFC")
    x = np.zeros((125, 125))
    start = time.time()
    y = model.predict([x])
    print(y)
    print(f"Time elapsed: {time.time()-start:.2f}s")