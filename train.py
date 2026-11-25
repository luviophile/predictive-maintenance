import numpy as np
import torch
from data import load_data, inspect_data, analyze_failures, split_data, print_report, split_validation
from preprocessing import scale_features, select_features, encode_type,NUMERIC_COLUMNS
from model import train_baseline,train_torch_model,predict_torch_proba
from evaluation import find_best_threshold, predict_with_threshold, evaluate_model
from persistence import save_artifacts


def main():
    df = load_data()
    data_info = inspect_data(df)
    failure_stats = analyze_failures(df)
    print_report(data_info, failure_stats)
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_val, y_train, y_val = split_validation(X_train, y_train)
    X_train_encoded, X_val_encoded,X_test_encoded = encode_type(X_train,X_val, X_test)
    X_train_scaled,X_val_scaled, X_test_scaled , scaler= scale_features(X_train_encoded, X_val_encoded, X_test_encoded)
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
    baseline_probs = baseline_model.predict_proba(X_val_scaled)[:, 1]
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    baseline_threshold, baseline_cost = find_best_threshold(
        baseline_probs, y_val, thresholds
    )
    torch_threshold, torch_cost = find_best_threshold(
        torch_probs, y_val, thresholds
    )

    metadata = {
        "threshold": float(torch_threshold),
        "model_name": "FaultNet",
        "feature_columns": X.columns.tolist(),
        "encoded_columns" :X_train_scaled.columns.tolist(),
        "numeric_columns":NUMERIC_COLUMNS
    }
    save_artifacts(torch_model, scaler,metadata)

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