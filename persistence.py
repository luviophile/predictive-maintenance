"""保存训练产物，并在服务启动时重建模型与预处理流程。"""

import json
import torch
from pathlib import Path
import joblib
from model import FaultNet

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def save_artifacts(model, scaler, metadata):
    """将模型权重、标准化器和推理元数据写入 artifacts 目录。

    参数
        model（FaultNet）为训练完成的网络。
        scaler（StandardScaler）为使用训练集拟合的标准化器。
        metadata（dict）为可序列化为 JSON 的配置，包含分类阈值和有序特征列等信息。

    返回
        None，产物写入 persistence.py 所在目录下的 artifacts 子目录。
    """
    artifact_dir = Path(ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 保存 state_dict 便于加载时由代码重建网络。标准化器独立保存，供推理复用训练集统计量。
    torch.save(model.state_dict(), artifact_dir/"faultnet.pth")
    joblib.dump(scaler, artifact_dir/"scaler.joblib")
    # 将阈值和列顺序放在可读的 JSON 中，便于核对服务实际采用的推理配置。
    with open(artifact_dir/"metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def load_artifacts():
    """从 persistence.py 所在目录下的 artifacts 子目录加载推理所需的三项产物。

    返回
        tuple，依次为已切换到评估模式的 FaultNet、已拟合的 StandardScaler 和元数据字典。
    """
    model = FaultNet()
    artifact_dir = Path(ARTIFACT_DIR)
    # 先实例化同一网络结构，再将参数映射到 CPU，便于在无 GPU 的环境中部署。
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