import torch
import torch.nn as nn
import numpy as np
from numpy.typing import ArrayLike

class SimpleFC(nn.Module):
    def __init__(self, out_features):
        super(SimpleFC, self).__init__()
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Linear(125*125, 1024), # 15,625 -> 1024
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
        


class TiltPredictor:
    
    def load_model(self, model:torch.nn.Module, fname="best_model.pth", path="./saved_models/") -> nn.Module:
        model.load_state_dict(torch.load(path+fname, weights_only=False, map_location=self.DEVICE))
        return model
    
    def __init__(self, model_fname):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.DEVICE)

        self.model = SimpleFC(2)
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
            img = torch.from_numpy(img).flatten().float()
            tensors.append(img)
        x = torch.stack(tensors).to(self.DEVICE)

        predictions:torch.Tensor = self.model.forward(x)
        predictions = predictions.detach().cpu().numpy()
        return predictions


if __name__=="__main__":
    import time
    model = TiltPredictor("fc_4layers_1024batch_500epochs_50cosinescheduler_best_model.pth")
    x = np.zeros((125, 125))
    start = time.time()
    y = model.predict([x])
    print(y)
    print(f"Time elapsed: {time.time()-start:.2f}s")