"""计算二分类指标，并根据验证集上的误报和漏报成本搜索分类阈值。"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_model(y_test, y_pred):
    """计算准确率、精确率、召回率、F1 分数和混淆矩阵。

    参数
        y_test、y_pred（pandas.Series 或 numpy.ndarray）
            分别为真实与预测的 0/1 标签，均为长度为样本数的一维序列。

    返回
        dict，包含四项标量指标和一个固定为 2×2 的混淆矩阵。
        行对应真实类别，列对应预测类别，行列均按 [0, 1] 排列。只有一个类别时也保留两行两列。
    """
    # 故障比例较低时，准确率可能掩盖漏报。召回率和精确率分别反映漏报和误报的情况。
    metrics = {"accuracy": accuracy_score(y_test, y_pred),
               "precision": precision_score(y_test, y_pred),
               "recall": recall_score(y_test, y_pred),
               "f1": f1_score(y_test, y_pred),
               "confusion_matrix": confusion_matrix(
                    y_test, y_pred, labels=[0, 1]
                )}
    return metrics


def predict_with_threshold(failure_probs, threshold):
    """按阈值将故障概率批量转换为类别标签。

    参数
        failure_probs（numpy.ndarray）为形状为（样本数，）的故障概率。
        threshold（float）为分类阈值，等于阈值的样本也判为故障。

    返回
        numpy.ndarray，形状与概率数组一致，元素 1 表示故障，0 表示正常。
    """
    y_pred = (failure_probs>= threshold).astype(int)
    return y_pred


def find_best_threshold(failure_probs, y_val, thresholds):
    """在验证集上遍历候选阈值，选择假设错误成本最低的一项。

    参数
        failure_probs（numpy.ndarray）为与验证标签同顺序的一维故障概率。
        y_val（pandas.Series 或 numpy.ndarray）为验证集的一维 0/1 标签。
        thresholds（可迭代对象）依次提供待比较的浮点阈值。

    返回
        tuple，包含最佳候选阈值与其总成本。同成本时保留先出现的阈值。
        无候选项时返回 None 和正无穷。
    """
    best_threshold = None
    best_cost = float('inf')
    for threshold in thresholds:
        y_pred= predict_with_threshold(failure_probs, threshold)
        # 固定行列的类别顺序，即使某一类没有出现，也能用稳定的下标取得误报数和漏报数。
        cm = confusion_matrix(y_val, y_pred,labels=[0, 1])
        fp = cm [0,1]
        fn = cm [1,0]
        # 实验假设漏报的单次成本是误报的十倍，通过决策阈值体现这种代价差异。
        total_cost =5000 * fp + 50000 * fn
        if total_cost < best_cost:
            best_threshold = threshold
            best_cost = total_cost
    return best_threshold, best_cost
