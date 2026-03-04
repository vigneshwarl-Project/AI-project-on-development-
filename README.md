

Hi 👋,

This is a small **statistics + ML metrics toolkit** that I built in pure Python,
without using libraries like NumPy or scikit-learn. The main goal is to
understand the formulas properly and see how these functions work internally.

## What is inside?

### Descriptive statistics
- `mean` – average value
- `median` – middle value after sorting
- `mode` – most frequent value
- `Variance` – how spread out the data is
- `std_dev` – standard deviation (spread in original units)
- `range_stat` – max - min
- `percentile` – value at a given percentage position (25th, 50th, 75th…)
- `quartiles` – Q1, Q2 (median), Q3
- `iqr` – interquartile range (Q3 - Q1)
- `z_scores` – standardize data to mean 0 and std 1
- `min_max_normalization` – scale values to [0, 1]
- `skewness` – checks if data is left/right skewed
- `kurtosis` – checks tail heaviness compared to normal distribution
- `covariance` – how two variables move together
- `correlation` – strength of linear relationship (-1 to 1)
- `mad` – mean absolute deviation

### Classification metrics (binary and multi-class)
- `confusion_matrix` – TP, TN, FP, FN counts
- `accuracy` – overall correctness
- `precision` – of predicted positives, how many are actually positive
- `recall` – of actual positives, how many we caught
- `f1_score` – balance between precision and recall
- `specificity` – true negative rate
- `balanced_accuracy` – average of recall and specificity
- `multi_confusion_matrix` – confusion matrix for multiple classes
- `multi_class_metrics` – per-class, macro and micro averages

### Regression metrics
- `mse` – mean squared error
- `rmse` – root mean squared error
- `mae` – mean absolute error
- `r2_score` – how much variance is explained by the model

### Extra utilities
- Vector operations: `vec_add`, `vec_sub`, `Scalar_multiplication`, `dot`
- Distances: `euclidean_distance`, `manhattan_distance`
- Similarity: `cosine_similarity`
- ROC & AUC: `roc_curve_points`, `auc_score`
- Ranking: `rank_data`

## Example usage

```python
from stats_toolkit import *

data =[1][2][3][4][5]

print("Mean:", mean(data))
print("Std Dev:", std_dev(data))
print("Z-scores:", z_scores(data))
print("Quartiles:", quartiles(data))
print("IQR:", iqr(data))

y_true =[6]
y_pred =[6]

print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
print("Accuracy:", accuracy(y_true, y_pred))
print("Precision:", precision(y_true, y_pred))
print("Recall:", recall(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))





# I built this
To practice Python using real math/statistics problems.
To clearly see how common metrics in machine learning work internally.
To have a reusable small library for future projects and teaching.
Feel free to fork, suggest improvements, or use parts of this in your own projects.


---like  oru small statistics toolkit Python la from scratch build pannirukken.
No external libraries use panna la, mostly math mattum. and
little help from (chatgpt )-- to  understanding  the concept  but logic mine
Idea enna na, library functions blind ah use pannaama, formula oda working clear ah purinjikanum nu.
File la basic stats ellam irukku – mean, median, mode, variance, standard deviation, range, percentiles,
 quartiles, IQR, z-scores, min-max normalization laam.
Idhukku mela covariance, correlation, skewness, kurtosis maadhiri slightly advance functions um
implement pannirukken to understand data distribution and relationships.
Machine learning side la, binary and multi-class classification metrics add pannirukken –
confusion_matrix, accuracy, precision, recall, F1-score and so on.
Regression projects ku MSE, RMSE, MAE, R2-score functions um irukku.
Ella function um simple Python loops and formulas la implement pannirukken,
 comments vechu easy ah follow panna maadhiri panirukken.
Small example script um add pannirukken so anyone can directly run and see outputs.
Idha future la projects la reuse pannalam,
 students kum concept explain pannara time la use pannalam nu nenachen.
Feedback kudutha romba useful ah irukkum sir –
enna improve panna mudiyum nu sollunga, na modify panraen.                                                                                                           


