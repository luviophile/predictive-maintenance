"""定义训练与推理共享的特征选择、类别编码和数值标准化规则。"""

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
    """提取推理时可获得的设备特征和故障标签。

    参数
        df（pandas.DataFrame）为包含原始设备字段和 Machine failure 标签的数据表。

    返回
        tuple，包含形状为（样本数，6）的特征表和长度为样本数的标签序列。
        六个输入字段由一个设备类型和五个数值字段组成。
    """
    # 显式选取输入字段，避免将设备标识、故障结果和故障子类型带入模型。
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
    """仅使用训练集拟合标准化器，随后转换三个数据集中的数值特征。

    标准化减小温度、转速等特征之间的尺度差异，便于模型优化。

    参数
        X_train_encoded、X_val_encoded、X_test_encoded（pandas.DataFrame）
            分别为训练集、验证集和测试集的编码结果，三者使用相同的列及列顺序。

    返回
        tuple，依次为三个标准化后的特征表及拟合好的 StandardScaler。
        特征表形状保持不变，独热编码列保留 0/1 值。
    """
    scaler = StandardScaler()
    X_train_scaled = X_train_encoded.copy()
    X_val_scaled = X_val_encoded.copy()
    X_test_scaled = X_test_encoded.copy()
    # fit_transform 从训练集学习均值和缩放系数。其余数据只调用 transform，避免泄漏统计信息。
    X_train_scaled[NUMERIC_COLUMNS] = scaler.fit_transform(X_train_scaled[NUMERIC_COLUMNS])
    X_val_scaled[NUMERIC_COLUMNS] = scaler.transform(X_val_scaled[NUMERIC_COLUMNS])
    X_test_scaled[NUMERIC_COLUMNS] = scaler.transform(X_test_scaled[NUMERIC_COLUMNS])
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def encode_type(X_train, X_val, X_test):
    """对设备类型做独热编码，并以训练集编码结果作为列模板。

    参数
        X_train、X_val、X_test（pandas.DataFrame）
            分别为含 Type 列的训练、验证和测试特征表。

    返回
        tuple，依次为编码后的训练、验证和测试特征表。
        当训练集包含 H、L、M 三种类型时，各表有五个数值列和三个类型列。
    """
    # 每种设备类型使用一个 0/1 指示列，避免给无序类别引入人为的大小关系。
    X_train_encoded = pd.get_dummies(X_train, columns=["Type"] ,dtype= int)
    X_val_encoded = pd.get_dummies(X_val, columns=["Type"] ,dtype= int)
    X_test_encoded = pd.get_dummies(X_test, columns=["Type"] ,dtype= int)
    # reindex 按训练列顺序重排，缺少的编码列补 0，模板外的列丢弃。
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
    """按已发布模型的特征规格预处理一批原始设备记录。

    参数
        raw_df（pandas.DataFrame）为待推理数据，每行对应一条设备记录。
        scaler（StandardScaler）为训练阶段拟合并保存的数值标准化器。
        metadata（dict）记录原始字段、编码后的有序列名和需要标准化的数值列。

    返回
        pandas.DataFrame，形状为（样本数，编码特征数），列顺序与训练时一致。
        当前发布的模型需要八列特征。

    异常
        ValueError，在缺少必需字段或出现不支持的设备类型时抛出。
    """
    required_columns=metadata["feature_columns"]
    missing_columns = [col for col in required_columns if col not in raw_df.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"缺少必需字段: {missing_columns}")
    raw_features = raw_df[required_columns].copy()
    # isin 逐行标记合法类型，取反后 any 可判断批次内是否存在非法记录。
    invalid_types = ~raw_features["Type"].isin(["L", "M", "H"])
    if invalid_types.any():
        raise ValueError("存在不支持的设备类型")
    encoded_df = pd.get_dummies(
        raw_features,
        columns=["Type"],
        dtype=int
    )
    # 单条请求通常只产生一个类型列，使用元数据补齐其余类型列并恢复训练时的顺序。
    encoded_df = encoded_df.reindex(
        columns=metadata["encoded_columns"],
        fill_value=0
    )
    processed_df = encoded_df.copy()

    numeric_columns = metadata["numeric_columns"]

    # 复用训练集统计量，确保同一个输入不会因推理批次的组成而改变缩放结果。
    processed_df[numeric_columns] = scaler.transform(
        processed_df[numeric_columns]
    )
    return processed_df