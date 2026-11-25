from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def load_data():
    dataset = fetch_ucirepo(id=601)
    df = dataset.data.original
    return df

def inspect_data(df):
    result = {
        "rows_count" :  df.shape[0],
        "columns_count" : df.shape[1],
        "field_names" : df.columns,
        "field_dtypes": df.dtypes,
        "missing_count" : df.isna().sum()
    }
    return result

def analyze_failures(df):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    return X_train, X_test, y_train, y_test

def split_validation(X_train_full, y_train_full):
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42,stratify=y_train_full)
    return X_train, X_val, y_train, y_val