# %% [markdown]
# # Decision Tree Classifier — Breast Cancer Wisconsin
#
# Build a `DecisionTreeClassifier`, evaluate it with a confusion matrix and tree plot,
# then compare hyperparameters one at a time using **5-fold cross-validation**.

# %%
import pandas as pd  # For Reading the data
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # To build our Decision tree
from sklearn.tree import plot_tree  # To show the tree
from sklearn.model_selection import train_test_split  # To split the data for testing and training
from sklearn.model_selection import cross_val_score  # To perform a K-fold Cross validation (K = 5 for SciKitLearn)
from sklearn.metrics import confusion_matrix  # To build a confusion matrix for our tree
from sklearn.metrics import ConfusionMatrixDisplay  # To plot the confusion matrix
import matplotlib.pyplot as plt

RANDOM_STATE = 42
CV_FOLDS = 5

# %% [markdown]
# ## 1. Load & Explore Data

# %%
# Breast Cancer Wisconsin (Diagnostic) — read with pandas
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)
df = pd.read_csv(url, header=None)

df.columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
df["target"] = df["diagnosis"].map({"M": 0, "B": 1})  # 0 = malignant, 1 = benign

print(df.shape)
print(df["diagnosis"].value_counts())
df.head()

# %%
feature_cols = [c for c in df.columns if c.startswith("feature_")]
X = df[feature_cols].values
y = df["target"].values
class_names = ["malignant", "benign"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# %% [markdown]
# ## 2. Train a Decision Tree (baseline)

# %%
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree.fit(X_train, y_train)

train_acc = tree.score(X_train, y_train)
test_acc = tree.score(X_test, y_test)
cv_scores = cross_val_score(tree, X_train, y_train, cv=CV_FOLDS)

print(f"Train accuracy:      {train_acc:.4f}")
print(f"Test accuracy:       {test_acc:.4f}")
print(f"CV accuracy (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# %% [markdown]
# ## 3. Visualize the Tree

# %%
plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=feature_cols,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree (baseline, full depth)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Confusion Matrix (test set)

# %%
y_pred = tree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix — Baseline Tree (Test Set)")
plt.tight_layout()
plt.show()

print("Confusion matrix:\n", cm)

# %% [markdown]
# ## 5. Hyperparameter Comparison (one at a time)
#
# `cross_val_score` runs 5-fold CV on the training split.
# Train metrics come from refitting on the full training set for each setting.

# %%
def log_loss_from_proba(y_true, y_proba):
    """Binary/multiclass log loss using predicted probabilities."""
    y_true = np.asarray(y_true)
    p = y_proba[np.arange(len(y_true)), y_true]
    return -np.mean(np.log(p + 1e-15))


def evaluate_hyperparameter(param_name, param_values, x_labels=None):
    rows = []
    for value in param_values:
        params = {param_name: value, "random_state": RANDOM_STATE}
        clf = DecisionTreeClassifier(**params)

        val_acc_scores = cross_val_score(
            clf, X_train, y_train, cv=CV_FOLDS, scoring="accuracy"
        )
        val_ll_scores = cross_val_score(
            clf, X_train, y_train, cv=CV_FOLDS, scoring="neg_log_loss"
        )

        clf.fit(X_train, y_train)
        train_proba = clf.predict_proba(X_train)

        rows.append(
            {
                param_name: value,
                "train_accuracy": clf.score(X_train, y_train),
                "val_accuracy": val_acc_scores.mean(),
                "val_accuracy_std": val_acc_scores.std(),
                "train_log_loss": log_loss_from_proba(y_train, train_proba),
                "val_log_loss": -val_ll_scores.mean(),
                "val_log_loss_std": val_ll_scores.std(),
            }
        )

    return pd.DataFrame(rows), x_labels


def plot_hyperparameter_results(df, param_name, title, x_labels=None):
    x = np.arange(len(df))
    labels = x_labels if x_labels is not None else [str(v) for v in df[param_name]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].errorbar(
        x, df["train_accuracy"], marker="o", label="Train", capsize=3
    )
    axes[0].errorbar(
        x,
        df["val_accuracy"],
        yerr=df["val_accuracy_std"],
        marker="s",
        label="CV Validation (5-fold)",
        capsize=3,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_xlabel(param_name.replace("_", " ").title())
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.85, 1.02)
    axes[0].legend()
    axes[0].set_title("Accuracy vs Hyperparameter")

    axes[1].plot(x, df["train_log_loss"], marker="o", label="Train")
    axes[1].errorbar(
        x,
        df["val_log_loss"],
        yerr=df["val_log_loss_std"],
        marker="s",
        label="CV Validation (5-fold)",
        capsize=3,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_xlabel(param_name.replace("_", " ").title())
    axes[1].set_ylabel("Log Loss")
    axes[1].legend()
    axes[1].set_title("Log Loss vs Hyperparameter")

    plt.tight_layout()
    plt.show()
    return fig

# %% [markdown]
# ### Criterion — Gini vs Entropy

# %%
df_criterion, _ = evaluate_hyperparameter("criterion", ["gini", "entropy"])
plot_hyperparameter_results(df_criterion, "criterion", "Split Criterion: Gini vs Entropy")
df_criterion.round(4)

# %% [markdown]
# ### Max Depth

# %%
max_depth_values = [1, 2, 3, 4, 5, 7, 10, 15, 20, None]
depth_labels = ["1", "2", "3", "4", "5", "7", "10", "15", "20", "None"]
df_max_depth, _ = evaluate_hyperparameter("max_depth", max_depth_values, depth_labels)
plot_hyperparameter_results(
    df_max_depth, "max_depth", "Max Depth", x_labels=depth_labels
)
df_max_depth.round(4)

# %% [markdown]
# ### Max Leaf Nodes

# %%
max_leaf_values = [2, 4, 8, 16, 32, 64, None]
leaf_labels = ["2", "4", "8", "16", "32", "64", "None"]
df_max_leaf, _ = evaluate_hyperparameter("max_leaf_nodes", max_leaf_values, leaf_labels)
plot_hyperparameter_results(
    df_max_leaf, "max_leaf_nodes", "Max Leaf Nodes", x_labels=leaf_labels
)
df_max_leaf.round(4)

# %% [markdown]
# ### Min Samples Split

# %%
df_min_split, _ = evaluate_hyperparameter(
    "min_samples_split", [2, 5, 10, 20, 30, 50]
)
plot_hyperparameter_results(df_min_split, "min_samples_split", "Min Samples Split")
df_min_split.round(4)

# %% [markdown]
# ### Min Samples Leaf

# %%
df_min_leaf, _ = evaluate_hyperparameter(
    "min_samples_leaf", [1, 2, 5, 10, 20, 30]
)
plot_hyperparameter_results(df_min_leaf, "min_samples_leaf", "Min Samples Leaf")
df_min_leaf.round(4)

# %% [markdown]
# ## 6. Final Model (tuned `max_depth` from CV)

# %%
best_depth = df_max_depth.loc[df_max_depth["val_accuracy"].idxmax(), "max_depth"]
best_depth = int(best_depth) if best_depth is not None and not pd.isna(best_depth) else None

final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
final_tree.fit(X_train, y_train)

final_train = final_tree.score(X_train, y_train)
final_test = final_tree.score(X_test, y_test)
final_cv = cross_val_score(final_tree, X_train, y_train, cv=CV_FOLDS)

print(f"Best max_depth from CV: {best_depth}")
print(f"Final train accuracy:   {final_train:.4f}")
print(f"Final test accuracy:    {final_test:.4f}")
print(f"Final CV accuracy:      {final_cv.mean():.4f} (+/- {final_cv.std():.4f})")

# %% [markdown]
# ### Final Tree Plot

# %%
plt.figure(figsize=(14, 8))
plot_tree(
    final_tree,
    feature_names=feature_cols,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title(f"Final Decision Tree (max_depth={best_depth})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Final Confusion Matrix

# %%
y_final_pred = final_tree.predict(X_test)
cm_final = confusion_matrix(y_test, y_final_pred)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm_final, display_labels=class_names).plot(ax=ax, cmap="Blues")
ax.set_title(f"Confusion Matrix — Final Tree (max_depth={best_depth})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Summary

# %%
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline test accuracy: {test_acc:.4f}")
print(f"Final test accuracy:    {final_test:.4f} (max_depth={best_depth})")
print()
print("Best CV validation accuracy per hyperparameter:")
for name, frame, col in [
    ("criterion", df_criterion, "criterion"),
    ("max_depth", df_max_depth, "max_depth"),
    ("max_leaf_nodes", df_max_leaf, "max_leaf_nodes"),
    ("min_samples_split", df_min_split, "min_samples_split"),
    ("min_samples_leaf", df_min_leaf, "min_samples_leaf"),
]:
    best = frame.loc[frame["val_accuracy"].idxmax()]
    print(f"  {name:18s} -> {best[col]}  (val acc = {best['val_accuracy']:.4f})")
