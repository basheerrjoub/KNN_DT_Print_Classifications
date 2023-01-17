from sklearn.model_selection import train_test_split

# Load the data into a Pandas DataFrame
import pandas as pd

df = pd.read_csv("train.csv")

# Split the data into a training set and a test set
X = df[["Obese", "like_desserts", "family_history"]]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.neighbors import KNeighborsClassifier

# Create a KNN model with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the training data
knn.fit(X_train, y_train)


import matplotlib.pyplot as plt

# Create a scatter plot of the data, with different classes being represented by different colors
plt.scatter(X_train["Obese"], X_train["like_desserts"], c=y_train, cmap="viridis")

# Add a legend
plt.colorbar()
plt.show()
