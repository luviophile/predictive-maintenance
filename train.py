"""编排数据分析、模型训练、阈值选择、产物保存和最终测试集评估。"""

import numpy as np
import torch
from data import load_data, inspect_data, analyze_failures, split_data, print_report, split_validation
from preprocessing import scale_features, select_features, encode_type,NUMERIC_COLUMNS
from model import train_baseline,train_torch_model,predict_torch_proba
from evaluation import find_best_threshold, predict_with_threshold, evaluate_model
from persistence import save_artifacts


def main():
    """训练逻辑回归和 FaultNet，并保存供 API 使用的 FaultNet 推理产物。

    返回
        None，训练产物写入 artifacts，阈值、业务成本和测试指标输出到控制台。
    """
    df = load_data()
    data_info = inspect_data(df)
    failure_stats = analyze_failures(df)
    print_report(data_info, failure_stats)
    X, y = select_features(df)
    # 先留出最终测试集，再从剩余样本分出验证集。训练、阈值选择和最终评估各自使用对应的数据。
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_val, y_train, y_val = split_validation(X_train, y_train)
    X_train_encoded, X_val_encoded,X_test_encoded = encode_type(X_train,X_val, X_test)
    X_train_scaled,X_val_scaled, X_test_scaled , scaler= scale_features(X_train_encoded, X_val_encoded, X_test_encoded)
    # 特征张量形状为（训练样本数，8），标签从一维转成（训练样本数，1）以匹配网络输出。
    X_train_tensor = torch.tensor(
        X_train_scaled.to_numpy(dtype=np.float32)
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(dtype=np.float32)
    ).reshape(-1, 1)
    baseline_model = train_baseline(X_train_scaled, y_train)

    torch_model, loss_history = train_torch_model(
        X_train_tensor,
        y_train_tensor,
        epochs=30,
        batch_size=64
    )
    torch_probs = predict_torch_proba(torch_model, X_val_scaled)
    # 在 0/1 标签下，逻辑回归概率矩阵的第 2 列对应故障类别。
    baseline_probs = baseline_model.predict_proba(X_val_scaled)[:, 1]
    # 两个模型使用同一验证集和同一组候选阈值，在分类决策阶段体现误报与漏报的成本差异。
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    baseline_threshold, baseline_cost = find_best_threshold(
        baseline_probs, y_val, thresholds
    )
    torch_threshold, torch_cost = find_best_threshold(
        torch_probs, y_val, thresholds
    )

    # 固定导出 FaultNet，并记录从原始字段到有序模型输入的映射及验证集选出的阈值。
    metadata = {
        "threshold": float(torch_threshold),
        "model_name": "FaultNet",
        "feature_columns": X.columns.tolist(),
        "encoded_columns" :X_train_scaled.columns.tolist(),
        "numeric_columns":NUMERIC_COLUMNS
    }
    save_artifacts(torch_model, scaler,metadata)

    # 在独立测试集上沿用已确定的阈值，度量该决策在未参与调参的样本上的表现。
    torch_test_probs = predict_torch_proba(
        torch_model, X_test_scaled
    )

    torch_test_pred = predict_with_threshold(
        torch_test_probs, torch_threshold
    )

    test_metrics = evaluate_model(
        y_test, torch_test_pred
    )

    cm = test_metrics["confusion_matrix"]

    test_cost = int(
        5000 * cm[0, 1] +
        50000 * cm[1, 0]
    )
    print("逻辑回归最佳阈值：", baseline_threshold)
    print("逻辑回归最低成本：", baseline_cost)
    print("PyTorch 最佳阈值：", torch_threshold)
    print("PyTorch 最低成本：", torch_cost)
    print("最终测试集评估：", test_metrics)
    print("最终测试集错误成本：", test_cost)


if __name__ == "__main__":
    main()