import math

# ------------------------------
# Basic Validation & Helpers
# ------------------------------

def validate_data(data):
    if not data:
        raise ValueError("Data cannot be empty")

def mean(data):
    validate_data(data)
    return sum(data) / len(data)

def Variance(data):
    mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / len(data)

def std_dev(data):
    return math.sqrt(Variance(data))

def maximum(data):
    validate_data(data)
    max_val = data[0]
    for x in data:
        if x > max_val:
            max_val = x
    return max_val

def minimum(data):
    validate_data(data)
    min_val = data[0]
    for x in data:
        if x < min_val:
            min_val = x
    return min_val

def custom_sum(data):
    total = 0
    for x in data:
        total += x
    return total

def vec_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def vec_sub(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def Scalar_multiplication(scalar, vector):
    return [scalar * x for x in vector]

def dot(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same length")
    return sum(a * b for a, b in zip(v1, v2))

def vector_norm(v):
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(v1, v2):
    num = dot(v1, v2)
    den = vector_norm(v1) * vector_norm(v2)
    if den == 0:
        return 0
    return num / den

# ------------------------------
# Z-scores, Percentiles, Quartiles, IQR
# ------------------------------

def z_scores(data):
    mu = mean(data)
    sigma = std_dev(data)
    if sigma == 0:
        return [0 for _ in data]
    return [(x - mu) / sigma for x in data]

def percentile(data, p):
    validate_data(data)
    data = sorted(data)
    n = len(data)
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return data[lower]
    if index == lower:
        return data[lower]
    fraction = index - lower
    return data[lower] + fraction * (data[upper] - data[lower])

def quartiles(data):
    Q1 = percentile(data, 25)
    Q2 = percentile(data, 50)
    Q3 = percentile(data, 75)
    return Q1, Q2, Q3

def iqr(data):
    Q1, _, Q3 = quartiles(data)
    return Q3 - Q1

def min_max_normalization(data):
    validate_data(data)
    min_val = minimum(data)
    max_val = maximum(data)
    if max_val == min_val:
        return [0 for _ in data]
    return [(x - min_val) / (max_val - min_val) for x in data]

# ------------------------------
# Extra Descriptive Statistics
# ------------------------------

def mode(data):
    validate_data(data)
    counts = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    if len(modes) == 1:
        return modes[0]
    return modes  # can be multimodal

def median(data):
    validate_data(data)
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2

def skewness(data):
    # population skewness: m3 / sigma^3
    validate_data(data)
    n = len(data)
    mu = mean(data)
    sigma = std_dev(data)
    if sigma == 0:
        return 0
    m3 = sum((x - mu) ** 3 for x in data) / n
    return m3 / (sigma ** 3)  # standardized third moment [web:1][web:14]

def kurtosis(data):
    # excess kurtosis: m4 / sigma^4 - 3
    validate_data(data)
    n = len(data)
    mu = mean(data)
    sigma = std_dev(data)
    if sigma == 0:
        return 0
    m4 = sum((x - mu) ** 4 for x in data) / n
    return m4 / (sigma ** 4) - 3  # standardized fourth moment minus 3 [web:6][web:12]

def mad(data):
    # mean absolute deviation around mean
    validate_data(data)
    mu = mean(data)
    return sum(abs(x - mu) for x in data) / len(data)

def range_stat(data):
    return maximum(data) - minimum(data)

# ------------------------------
# Covariance & Correlation
# ------------------------------

def covariance(x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    n = len(x)
    if n == 0:
        raise ValueError("Data cannot be empty")
    mx = mean(x)
    my = mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n

def correlation(x, y):
    cov = covariance(x, y)
    sx = math.sqrt(covariance(x, x))
    sy = math.sqrt(covariance(y, y))
    if sx == 0 or sy == 0:
        return 0
    return cov / (sx * sy)

# ------------------------------
# Ranking utilities
# ------------------------------

def rank_data(data):
    # average ranks for ties
    indexed = list(enumerate(data))
    indexed.sort(key=lambda t: t[1])  # sort by value
    ranks = [0.0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranks then average
        for k in range(i, j + 1):
            original_index = indexed[k][0]
            ranks[original_index] = avg_rank
        i = j + 1
    return ranks

# ------------------------------
# Classification Metrics (Binary)
# ------------------------------

def confusion_matrix(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")
    TP = TN = FP = FN = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            TP += 1
        elif actual == 0 and predicted == 0:
            TN += 1
        elif actual == 0 and predicted == 1:
            FP += 1
        elif actual == 1 and predicted == 0:
            FN += 1
    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return (TP + TN) / len(y_true)

def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)

def specificity(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    if TN + FP == 0:
        return 0
    return TN / (TN + FP)

def error_rate(y_true, y_pred):
    return 1 - accuracy(y_true, y_pred)

def balanced_accuracy(y_true, y_pred):
    return (recall(y_true, y_pred) + specificity(y_true, y_pred)) / 2

# ------------------------------
# Multi-Class Metrics
# ------------------------------

def multi_confusion_matrix(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")
    classes = sorted(list(set(y_true)))
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    return matrix

def multi_class_metrics(y_true, y_pred):
    classes = sorted(list(set(y_true)))
    cm = multi_confusion_matrix(y_true, y_pred)
    metrics = {}
    macro_p = macro_r = macro_f1 = 0
    micro_TP = micro_FP = micro_FN = 0
    for c in classes:
        TP = cm[c][c]
        FP = sum(cm[other][c] for other in classes if other != c)
        FN = sum(cm[c][other] for other in classes if other != c)
        support = sum(cm[c].values())
        p = TP / (TP + FP) if (TP + FP) != 0 else 0
        r = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0
        metrics[c] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support
        }
        macro_p += p
        macro_r += r
        macro_f1 += f1
        micro_TP += TP
        micro_FP += FP
        micro_FN += FN
    macro_avg = {
        "precision": macro_p / len(classes),
        "recall": macro_r / len(classes),
        "f1": macro_f1 / len(classes)
    }
    micro_precision = micro_TP / (micro_TP + micro_FP) if (micro_TP + micro_FP) != 0 else 0
    micro_recall = micro_TP / (micro_TP + micro_FN) if (micro_TP + micro_FN) != 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) != 0 else 0
    micro_avg = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1
    }
    return metrics, macro_avg, micro_avg

# ------------------------------
# Regression Metrics
# ------------------------------

def mse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")
    return sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / len(y_true)

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")
    return sum(abs(a - p) for a, p in zip(y_true, y_pred)) / len(y_true)

def r2_score(y_true, y_pred):
    validate_data(y_true)
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")
    mean_y = mean(y_true)
    ss_total = sum((y - mean_y) ** 2 for y in y_true)
    ss_residual = sum((a - p) ** 2 for a, p in zip(y_true, y_pred))
    if ss_total == 0:
        return 0
    return 1 - ss_residual / ss_total

# ------------------------------
# Probability & Loss functions
# ------------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(vector):
    exp_vals = [math.exp(x) for x in vector]
    total = sum(exp_vals)
    if total == 0:
        return [0 for _ in vector]
    return [v / total for v in exp_vals]

def binary_log_loss(y_true, y_prob):
    if len(y_true) != len(y_prob):
        raise ValueError("Length of y_true and y_prob must be equal")
    epsilon = 1e-15
    loss = 0.0
    for y, p in zip(y_true, y_prob):
        p = max(min(p, 1 - epsilon), epsilon)
        loss += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -loss / len(y_true)

def cross_entropy(y_true, y_prob):
    # y_true: one-hot, y_prob: predicted probabilities
    if len(y_true) != len(y_prob):
        raise ValueError("Length of y_true and y_prob must be equal")
    epsilon = 1e-15
    loss = 0.0
    for i in range(len(y_true)):
        if len(y_true[i]) != len(y_prob[i]):
            raise ValueError("Row length mismatch in y_true and y_prob")
        for j in range(len(y_true[i])):
            p = max(min(y_prob[i][j], 1 - epsilon), epsilon)
            loss += y_true[i][j] * math.log(p)
    return -loss / len(y_true)

# ------------------------------
# Gini, Entropy, ROC, AUC
# ------------------------------

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    impurity = 1.0
    for c in counts.values():
        p = c / n
        impurity -= p ** 2  # 1 - sum(p_k^2) [web:10][web:13]
    return impurity

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)  # -sum p_k log2 p_k [web:10][web:13]
    return ent

def roc_curve_points(y_true, y_scores):
    if len(y_true) != len(y_scores):
        raise ValueError("Length of y_true and y_scores must be equal")
    thresholds = sorted(set(y_scores), reverse=True)
    points = []
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_scores]
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        points.append((FPR, TPR))
    # Always include (0,0) and (1,1) for robustness
    points.append((0.0, 0.0))
    points.append((1.0, 1.0))
    # Sort by FPR
    points = sorted(set(points))
    return points

def auc_score(roc_points):
    roc_points = sorted(roc_points)
    area = 0.0
    for i in range(1, len(roc_points)):
        x1, y1 = roc_points[i - 1]
        x2, y2 = roc_points[i]
        area += (x2 - x1) * (y1 + y2) / 2.0
    return area

# ------------------------------
# Correlation distance / similarity
# ------------------------------

def euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same length")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def manhattan_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same length")
    return sum(abs(a - b) for a, b in zip(v1, v2))

# ------------------------------
# Quick Demo (you can keep or delete)
# ------------------------------

if __name__ == "__main__":
    data = [10, 12, 14, 16, 18]
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]

    print("Mean:", mean(data))
    print("Variance:", Variance(data))
    print("Standard Deviation:", std_dev(data))
    print("Maximum:", maximum(data))
    print("Minimum:", minimum(data))
    print("Custom Sum:", custom_sum(data))
    print("Vector Add:", vec_add(v1, v2))
    print("Vector Subtract:", vec_sub(v1, v2))
    print("Scalar Multiply:", Scalar_multiplication(3, v1))
    print("Dot:", dot(v1, v2))
    print("Cosine Similarity:", cosine_similarity(v1, v2))
    print("Z-Scores:", z_scores(data))
    print("25th Percentile:", percentile(data, 25))
    print("50th Percentile:", percentile(data, 50))
    print("75th Percentile:", percentile(data, 75))
    Q1, Q2, Q3 = quartiles(data)
    print("Quartiles:", Q1, Q2, Q3)
    print("IQR:", iqr(data))
    print("Mode:", mode(data))
    print("Median:", median(data))
    print("Range:", range_stat(data))
    print("Skewness:", skewness(data))
    print("Kurtosis (excess):", kurtosis(data))
    print("MAD:", mad(data))

    # Classification test
    y_true = [1, 0, 1, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
    print("Accuracy:", accuracy(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Specificity:", specificity(y_true, y_pred))
    print("Balanced Accuracy:", balanced_accuracy(y_true, y_pred))
    print("Error Rate:", error_rate(y_true, y_pred))

    # Regression test
    y_true_reg = [3, -0.5, 2, 7]
    y_pred_reg = [2.5, 0.0, 2, 8]
    print("\nRegression Metrics:")
    print("MSE:", mse(y_true_reg, y_pred_reg))
    print("RMSE:", rmse(y_true_reg, y_pred_reg))
    print("MAE:", mae(y_true_reg, y_pred_reg))
    print("R2 Score:", r2_score(y_true_reg, y_pred_reg))

    # ROC / AUC test
    y_scores = [0.9, 0.2, 0.8, 0.4, 0.3, 0.95, 0.6, 0.1]
    roc_pts = roc_curve_points(y_true, y_scores)
    print("\nROC Points:", roc_pts)
    print("AUC:", auc_score(roc_pts))

    # Correlation / distances
    x = [1, 2, 3, 4, 5]
    y = [2, 1, 4, 3, 5]
    print("\nCovariance:", covariance(x, y))
    print("Correlation:", correlation(x, y))
    print("Euclidean distance:", euclidean_distance(v1, v2))
    print("Manhattan distance:", manhattan_distance(v1, v2))
