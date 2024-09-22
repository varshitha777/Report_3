# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Convert continuous target to discrete classes (e.g., low, medium, high)
# Here, we define thresholds for classification
bins = [0, 1.5, 2.5, 5.0]  # Define bins based on your needs
labels = [0, 1, 2]  # Corresponding labels for each bin
y_class = np.digitize(y, bins) - 1  # Convert to classes

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Random Forest Classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
model.fit(X_train_scaled, y_train)
# Predict
y_pred = model.predict(X_test_scaled)
# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Optional: Plotting the confusion matrix
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_class)))
plt.xticks(tick_marks, np.unique(y_class))
plt.yticks(tick_marks, np.unique(y_class))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
