import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("train.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = DecisionTreeClassifier(max_depth=2).fit(X, y)


from sklearn.tree import plot_tree

# Training the DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2).fit(X, y)

# Visualizing the Decision Tree
sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=df.columns[:-1],
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    fontsize=14,
)
plt.savefig("Decision_Tree.png")


# Visualizing the Decision Tree
sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))
tree = plot_tree(
    model,
    feature_names=df.columns[:-1],
    filled=True,
    fontsize=20,
)
plt.savefig("Decision_Tree.png")
