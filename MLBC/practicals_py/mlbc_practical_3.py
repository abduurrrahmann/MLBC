import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# import load_breast_cancer function from sklearn.datasets module
from sklearn.datasets import load_breast_cancer


# 1. Load the real dataset
# load_breast_cancer function creates dataset object and stores it in 'cancer'.
# It's a standard scikit-learn dataset object, it's organized as follows:-
#  - .data: The main data (the features or X values).
#  - .target: The "answer key" (the labels or y values).
cancer = load_breast_cancer()


# 2. Prepare the data
# From the cancer dataset, we access the features using '.data'.
# '[:, 0]' selects all rows but only from the first column ('mean radius').
# '.reshape(-1, 1)' then turns this data into a single vertical column, which is the required format for scikit-learn models.
# 1 means you want one column.
# -1 is a placeholder that tells NumPy to automatically calculate the number of rows needed.
X = cancer.data[:, 0].reshape(-1, 1)


# We access the list of labels (0 for malignant, 1 for benign) using '.target'.
y = cancer.target


# 3. Create and train the model
model = LogisticRegression()
model.fit(X, y)


# 4. Create a range of values to plot the logistic curve smoothly
# 'np.linspace()' is a NumPy function that creates an array of evenly spaced numbers.
# We create 300 points between the minimum and maximum tumor sizes found in our data.
# This creates a smooth set of points on which to plot our prediction line.
X_test = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)


# Predict probabilities for the smooth range.
# 'model.predict_proba()' returns two columns of probabilities (for class 0 and class 1).
# The slice '[:, 1]' selects all rows (:) but only the second column (1),
# which contains the probabilities for class 1 (Benign).
y_prob = model.predict_proba(X_test)[:, 1]


# 5. Plot the results
plt.scatter(X, y, color='blue', label='Actual Data (0=Malignant, 1=Benign)', alpha=0.4)
plt.plot(X_test, y_prob, color='red', label='Logistic Regression Curve')
plt.xlabel('Mean Radius of Tumor')
plt.ylabel('Probability of Being Benign')
plt.title('Logistic Regression on Breast Cancer Data')
plt.legend()
plt.show()
