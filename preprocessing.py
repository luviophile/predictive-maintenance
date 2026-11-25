from sklearn.preprocessing import StandardScaler
import pandas as pd

NUMERIC_COLUMNS = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear"
    ]


def select_features(df):
    feature_columns = [
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear"
    ]
    X = df[feature_columns]
    y = df["Machine failure"]
    return X,y

def scale_features(X_train_encoded,X_val_encoded, X_test_encoded):
    scaler = StandardScaler()
    X_train_scaled = X_train_encoded.copy()
    X_val_scaled = X_val_encoded.copy()
    X_test_scaled = X_test_encoded.copy()
    X_train_scaled[NUMERIC_COLUMNS] = scaler.fit_transform(X_train_scaled[NUMERIC_COLUMNS])
    X_val_scaled[NUMERIC_COLUMNS] = scaler.transform(X_val_scaled[NUMERIC_COLUMNS])
    X_test_scaled[NUMERIC_COLUMNS] = scaler.transform(X_test_scaled[NUMERIC_COLUMNS])
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def encode_type(X_train, X_val, X_test):
    X_train_encoded = pd.get_dummies(X_train, columns=["Type"] ,dtype= int)
    X_val_encoded = pd.get_dummies(X_val, columns=["Type"] ,dtype= int)
    X_test_encoded = pd.get_dummies(X_test, columns=["Type"] ,dtype= int)
    X_val_encoded = X_val_encoded.reindex(
        columns=X_train_encoded.columns,
        fill_value= 0
    )
    X_test_encoded = X_test_encoded.reindex(
        columns=X_train_encoded.columns,
        fill_value= 0
    )
    return X_train_encoded, X_val_encoded, X_test_encoded

def preprocess_input(raw_df, scaler, metadata):
    required_columns=metadata["feature_columns"]
    missing_columns = [col for col in required_columns if col not in raw_df.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"缺少必需字段: {missing_columns}")
    raw_features = raw_df[required_columns].copy()
    invalid_types = ~raw_features["Type"].isin(["L", "M", "H"])
    if invalid_types.any():
        raise ValueError("存在不支持的设备类型")
    encoded_df = pd.get_dummies(
        raw_features,
        columns=["Type"],
        dtype=int
    )
    encoded_df = encoded_df.reindex(
        columns=metadata["encoded_columns"],
        fill_value=0
    )
    processed_df = encoded_df.copy()

    numeric_columns = metadata["numeric_columns"]

    processed_df[numeric_columns] = scaler.transform(
        processed_df[numeric_columns]
    )
    return processed_df