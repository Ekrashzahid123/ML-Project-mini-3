# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf

# Importing the dataset
dataset = pd.read_csv('loan_data.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

ct = ColumnTransformer([("loan_intent , previous_loan_defaults_on_file", OneHotEncoder(), [4,9])], remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]  # Avoid dummy variable trap

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train_Stand = sc.fit_transform(X_train)
X_test_Stand = sc.transform(X_test)

# Function to evaluate and save results
def evaluate_model(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print(f"\n{name} - Confusion Matrix:")
    print(cm)
    print(f"\n{name} - Classification Report:")
    print(report)
    print(f"{name} - Accuracy: {accuracy:.4f}\n")

    # Save confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

    # Save classification report as a text file
    with open(f"{name}_classification_report.txt", "w") as f:
        f.write(f"{name} - Classification Report\n")
        f.write(report)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train_Stand, y_train)
y_pred_knn = knn.predict(X_test_Stand)
evaluate_model("K-Nearest Neighbors", y_test, y_pred_knn)

# Support Vector Machine
svm = SVC(kernel='linear', random_state=0)
svm.fit(X_train_Stand, y_train)
y_pred_svm = svm.predict(X_test_Stand)
evaluate_model("Support Vector Machine", y_test, y_pred_svm)

# Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
evaluate_model("Decision Tree", y_test, y_pred_dt)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_Stand, y_train)
y_pred_nb = nb.predict(X_test_Stand)
evaluate_model("Naive Bayes", y_test, y_pred_nb)

# Artificial Neural Network
input_dim = X_train.shape[1]
ann = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='relu', input_dim=input_dim),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Single output for binary classification
])

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN
ann.fit(X_train_Stand, y_train, batch_size=10, epochs=10, verbose=0)

# Making predictions and evaluating ANN
y_pred_ann = (ann.predict(X_test_Stand) > 0.5).astype(int)
evaluate_model("Artificial Neural Network", y_test, y_pred_ann)
