import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#1. Load the Iris dataset

iris = load_iris()
X = iris.data
y = iris.target

#2. Split the dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3. Create the Decision Tree model

model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

#4. Train the model

model.fit(X_train, y_train)

#5. Predict the test data

y_pred = model.predict(X_test)

#6. Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)

#7. Display confusion matrix

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

#8. Visualize the Decision Tree

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Classifier")
plt.show()