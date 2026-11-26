"""实现逻辑回归基线、FaultNet 网络及其小批量训练和概率推理。"""

from sklearn.linear_model import LogisticRegression

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
import numpy as np

class FaultNet(nn.Module):
    """用于设备故障二分类的全连接网络。

    对于形状为（批大小，8）的输入，线性层依次将特征维度变为 16、8、1。
    隐藏层使用 ReLU 引入非线性，输出层为每条记录生成一个原始分数（logit）。
    """

    def __init__(self) -> None:
        """初始化网络并注册线性层参数，使其可被优化器和 state_dict 访问。"""
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向传播，输出供损失函数或推理使用的原始分数。

        参数
            x（torch.Tensor）为形状为（批大小，8）的浮点特征张量。

        返回
            torch.Tensor，形状为（批大小，1），每行是对应样本发生故障的 logit。
        """
        return self.model(x)

def train_baseline(X_train_scaled, y_train):
    """训练逻辑回归基线，为评估全连接网络提供对照。

    参数
        X_train_scaled（pandas.DataFrame）为编码并标准化后的训练特征表。
        y_train（pandas.Series）为与训练特征逐行对应的一维 0/1 标签。

    返回
        LogisticRegression，已拟合的基线模型。
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model


def train_torch_model(X_train_scaled, y_train, epochs, batch_size):
    """使用小批量梯度训练 FaultNet，并记录每轮的平均训练损失。

    参数
        X_train_scaled（torch.Tensor）为形状为（样本数，8）的 float32 特征张量。
        y_train（torch.Tensor）为形状为（样本数，1）的 float32 标签张量，取值为 0 或 1。
        epochs（int）为遍历训练数据的轮数。
        batch_size（int）为每批最多包含的样本数。

    返回
        tuple，包含已训练的 FaultNet 和按轮次排列的损失列表。
    """
    # 固定 PyTorch 随机种子，便于复现当前训练流程中的参数初始化和批次打乱。
    torch.manual_seed(42)
    model = FaultNet()
    # 损失函数在内部合并 Sigmoid 与二元交叉熵，直接接收 logit，数值上更稳定。
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

            # PyTorch 默认累加梯度，每批反向传播前清零，避免混入上一批的梯度。
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            # 反向传播求出各参数的梯度，Adam 随后根据梯度更新参数。
            loss.backward()
            optimizer.step()
            # 将批均值乘回实际批大小，使最后一个较小的批次也按样本数计入轮均值。
            epoch_loss += loss.item()*len(batch_X)
        loss_history.append(epoch_loss/len(dataset))
    return model ,loss_history


def predict_torch_proba(model, X_scaled):
    """计算每条设备记录属于故障类别的概率。

    参数
        model（FaultNet）为已训练的故障分类模型。
        X_scaled（pandas.DataFrame）为按训练列顺序排列的特征表，形状为（样本数，8）。

    返回
        numpy.ndarray，形状为（样本数，），顺序与输入行一致。
    """
    X_tensor = torch.tensor(
        X_scaled.to_numpy(dtype=np.float32)
    )
    model.eval()
    # eval 设置推理模式，no_grad 停止记录梯度。Sigmoid 将 logit 转成概率。
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits)
        # 将（样本数，1）展平成一维，便于与标签逐项对应和执行向量化阈值判断。
        probabilities = probabilities.cpu().numpy().reshape(-1)
    return probabilities
