import json
import torch
from pathlib import Path
import joblib
from model import FaultNet


def save_artifacts(model, scaler, metadata):
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir/"faultnet.pth")
    joblib.dump(scaler, artifact_dir/"scaler.joblib")
    with open(artifact_dir/"metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def load_artifacts():
    model = FaultNet()
    artifact_dir = Path("artifacts")
    state_dict = torch.load(
        artifact_dir/"faultnet.pth",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    scaler = joblib.load(artifact_dir/"scaler.joblib")
    with open(artifact_dir/"metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, metadata