from sklearn.linear_model import LogisticRegression

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
import numpy as np

class FaultNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def train_baseline(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model


def train_torch_model(X_train_scaled, y_train, epochs, batch_size):
    torch.manual_seed(42)
    model = FaultNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
    dataset = TensorDataset(X_train_scaled, y_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X,batch_y in loader:

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*len(batch_X)
        loss_history.append(epoch_loss/len(dataset))
    return model ,loss_history


def predict_torch_proba(model, X_scaled):
    X_tensor = torch.tensor(
        X_scaled.to_numpy(dtype=np.float32)
    )
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.cpu().numpy().reshape(-1)
    return probabilities
