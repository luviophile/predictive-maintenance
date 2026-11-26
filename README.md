
# 工业设备智能故障预测系统

Industrial Equipment Predictive Maintenance System

## 1. 项目介绍

本项目基于机器学习和深度学习技术，根据工业设备的运行参数进行故障分类与风险预警。

项目覆盖从数据分析、特征工程、模型训练、模型评估、分类阈值优化，到模型持久化、API 服务部署及自动化测试的完整 AI 工程流程。

## 2. 技术栈

- Python 3.12
- NumPy / Pandas
- Scikit-learn
- PyTorch
- FastAPI / Pydantic / Uvicorn
- Joblib
- Pytest / HTTPX
- Conda

## 3. 数据集

使用 UCI AI4I 2020 Predictive Maintenance Dataset。

数据集地址：https://doi.org/10.24432/C5HS5C

- 总样本数：10000
- 原始字段数：14
- 正常样本：9661
- 故障样本：339
- 故障率：3.39%

模型使用六个原始输入特征：

| 特征 | 含义 |
|---|---|
| Type | 设备类型 |
| Air temperature | 空气温度（K） |
| Process temperature | 工艺温度（K） |
| Rotational speed | 转速（rpm） |
| Torque | 扭矩（Nm） |
| Tool wear | 刀具磨损时间（min） |

预测目标为 Machine failure，即设备是否发生故障。

## 4. 数据预处理

- 使用分层抽样划分训练集、验证集和测试集。
- 数据集划分比例为 64% / 16% / 20%。
- 对设备类型进行 One-Hot 编码。
- 使用 StandardScaler 标准化五个数值特征。
- 仅使用训练集拟合标准化器，避免数据泄漏。
- 保存特征名称及排列顺序，保证训练与推理阶段的数据处理一致。

## 5. 模型设计

### 基准模型

使用 Logistic Regression 作为基准分类模型。

### 深度学习模型

使用 PyTorch 构建全连接神经网络 FaultNet。

网络结构：

```text
Input (8)
    ↓
Linear (8 → 16)
    ↓
ReLU
    ↓
Linear (16 → 8)
    ↓
ReLU
    ↓
Linear (8 → 1)
    ↓
Sigmoid（推理阶段）
```

训练配置：

- Loss：BCEWithLogitsLoss
- Optimizer：Adam
- Learning rate：0.001
- Batch size：64
- Epochs：30
- Random seed：42

## 6. 分类阈值与业务成本优化

针对故障样本占比较低的问题，引入基于假设业务成本的分类阈值优化策略。

假设：

- 单次误报成本（FP）：5000 元
- 单次漏报成本（FN）：50000 元

成本计算公式：

Total Cost = 5000 × FP + 50000 × FN

分别在验证集上对逻辑回归和 PyTorch FaultNet 进行候选分类阈值搜索，
比较两个模型在假设业务成本下的表现。

### 验证集模型对比

| 模型 | 最佳候选阈值 | 最低错误成本 |
|---|---:|---:|
| Logistic Regression | 0.1 | 1330000 |
| PyTorch FaultNet | 0.1 | 1180000 |

相比逻辑回归，PyTorch 模型的验证集假设错误成本降低约 11.28%。

根据本次验证集实验结果，人工选定 FaultNet 作为预测服务模型。
当前系统固定部署 FaultNet，暂未实现自动模型选择机制。

## 7. 最终测试集评估

在独立的 2000 条测试样本上进行评估。

| 评估指标 | 结果 |
|---|---:|
| Accuracy | 94.90% |
| Precision | 37.86% |
| Recall | 77.94% |
| F1-score | 50.96% |
| Classification Threshold | 0.1 |
| Total Error Cost | 1185000 |

混淆矩阵：

|  | 预测正常 | 预测故障 |
|---|---:|---:|
| 真实正常 | 1845 | 87 |
| 真实故障 | 15 | 53 |

模型成功识别了 68 条真实故障样本中的 53 条。

## 8. 系统架构

```text
原始工业设备参数
        ↓
FastAPI / Pydantic
        ↓
输入数据校验
        ↓
One-Hot 编码
        ↓
StandardScaler
        ↓
PyTorch FaultNet
        ↓
故障概率与阈值判断
        ↓
JSON 预测响应
```

使用 FastAPI Lifespan 管理模型生命周期，在应用启动时加载模型与预处理器，避免每次请求重复加载模型。

## 9. 项目目录

```text
predictive-maintenance/
├── data.py
├── preprocessing.py
├── model.py
├── evaluation.py
├── persistence.py
├── train.py
├── api.py
├── test_api.py
├── environment.yml
├── README.md
├── .gitignore
└── artifacts/
    ├── faultnet.pth
    ├── scaler.joblib
    └── metadata.json
```

## 10. 环境安装

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate predictive-maintenance
```

## 11. 模型训练

如需重新训练模型：

```bash
python train.py
```

训练程序将自动下载数据集、进行数据预处理、训练模型、优化分类阈值、评估模型并保存训练成果。

## 12. 启动预测服务

项目提供预训练模型，无需重新训练即可启动预测服务。

```bash
python -m uvicorn api:app --reload
```

Swagger API 文档：

http://127.0.0.1:8000/docs

健康检查接口：

GET /health

故障预测接口：

POST /predict

### 请求示例

```json
{
    "device_type": "L",
    "air_temperature": 300.2,
    "process_temperature": 310.5,
    "rotational_speed": 1500,
    "torque": 42.8,
    "tool_wear": 180
}
```

### 实际响应示例

```json
{
    "failure_probability": 0.01048144232481718,
    "predicted_failure": 0,
    "threshold": 0.1
}
```

predicted_failure 为 0 表示预测正常，为 1 表示预测故障。

## 13. 自动化测试

```bash
python -m pytest test_api.py -v
```

测试覆盖：

- 健康检查接口
- 常规设备预测
- 高风险设备预测
- 非法设备类型校验

当前开发环境测试结果：4 passed。

## 14. 项目局限性

- 使用合成数据集，尚未在真实工业设备上验证。
- 误报与漏报成本为实验假设，并非实际企业财务数据。
- 模型输出概率尚未经过概率校准。
- 当前模型根据输入参数进行故障分类，尚不具备基于时间序列的提前故障预测能力。
- 当前特征工程及 FaultNet 网络结构针对 AI4I 数据集的固定八维输入设计，
  更换数据集时需要重新确认类别编码与模型输入维度。
- 已在 Windows 下通过 environment.yml 创建独立 Conda 环境，
  成功复现模型训练、评估结果及全部 4 项 API 自动化测试。
  尚未验证其他操作系统的兼容性。

本项目为 AI 工程实践项目，不直接用于实际工业生产安全决策。
