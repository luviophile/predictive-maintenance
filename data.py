"""读取 UCI 设备数据，汇总字段与故障统计，并按标签分层划分数据。"""

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def load_data():
    """从 UCI 获取 AI4I 2020 预测性维护数据集。

    返回
        pandas.DataFrame，包含原始特征、标识字段及故障标签，每行对应一条设备记录。
    """
    dataset = fetch_ucirepo(id=601)
    df = dataset.data.original
    return df

def inspect_data(df):
    """汇总数据维度、字段名称、字段类型和逐列缺失值数量。

    参数
        df（pandas.DataFrame）为待检查的数据表，形状为（样本数，原始列数）。

    返回
        dict，包含行数、列数、字段索引，以及按字段索引的数据类型和缺失值数量。
    """
    result = {
        "rows_count" :  df.shape[0],
        "columns_count" : df.shape[1],
        "field_names" : df.columns,
        "field_dtypes": df.dtypes,
        "missing_count" : df.isna().sum()
    }
    return result

def analyze_failures(df):
    """计算正常与故障样本的数量及故障率，用于了解标签分布。

    参数
        df（pandas.DataFrame）为包含 Machine failure 列的数据表，0 表示正常，1 表示故障。

    返回
        dict，包含两个类别的计数和故障样本占总行数的比例。空表的故障率为 None。
    """
    counts = df["Machine failure"].value_counts()
    if df.shape[0] == 0:
        fault_rate =None
    else:
        fault_rate = counts.get(1,0)/df.shape[0]

    result = {
        "normal_record":counts.get(0,0),
        "fault_record":counts.get(1,0),
        "fault_rate":fault_rate
    }
    return result

def print_report(data_info, failure_stats):
    """将字段概况和故障分布输出到控制台。

    参数
        data_info（dict）为 inspect_data 返回的数据概况。
        failure_stats（dict）为 analyze_failures 返回的故障统计。

    返回
        None，报告通过标准输出呈现。
    """
    print("===== 工业设备数据分析报告 =====")
    print()
    print("数据总行数：", data_info["rows_count"])
    print("数据总列数：", data_info["columns_count"])
    print()
    print("===== 字段信息 =====")
    print("字段名称                  数据类型   缺失数量")
    for name in data_info["field_names"]:
       print(name,data_info["field_dtypes"][name],data_info["missing_count"][name])
    print()
    print("===== 设备故障统计 =====")
    print("正常记录数:", failure_stats["normal_record"])
    print("故障记录数:", failure_stats["fault_record"])
    if failure_stats["fault_rate"] is not None:
        print("故障率:", f"{failure_stats['fault_rate']:.2%}")
    else:
        print("无数据，失败率无法计算")
    print()
    print("===== 报告结束 =====")

def split_data(X, y):
    """划分建模用数据与独立测试集，分别约占总样本的 80% 和 20%。

    参数
        X（pandas.DataFrame）为形状为（样本数，特征数）的特征表。
        y（pandas.Series）为与 X 逐行对应的一维二分类标签。

    返回
        tuple，依次为建模用特征、测试特征、建模用标签和测试标签。
    """
    # 故障样本占比较低，按 y 分层可在两部分中近似保持故障比例。固定种子便于复现划分。
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    return X_train, X_test, y_train, y_test

def split_validation(X_train_full, y_train_full):
    """从建模用数据中分出验证集，供模型比较和阈值选择使用。

    参数
        X_train_full（pandas.DataFrame）为第一阶段划出的建模用特征表。
        y_train_full（pandas.Series）为与特征表逐行对应的标签。

    返回
        tuple，依次为训练特征、验证特征、训练标签和验证标签。
        两次划分后，训练集、验证集和测试集约占全量数据的 64%、16% 和 20%。
    """
    # 在建模用数据内部再次按标签分层，已留出的测试集独立用于最终评估。
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42,stratify=y_train_full)
    return X_train, X_val, y_train, y_val