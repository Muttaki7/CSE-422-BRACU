#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load the dataset
data_path = data_path = r'C:\Users\Muttaki\Downloads\bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)

# Preview the dataset
print("Dataset preview:")
print(data.head())

# Selecting features and label based on the dataset structure
feature_columns = [
   'Household_Total',
   'Population_Total',
   'Household_Excluding_Slum_Floating',
   'Population_Excluding_Slum_Floating'
]  # Features
label_column = 'Division'  # Label

# Filter the dataset to include only the selected features and label
selected_data = data[feature_columns + [label_column]]

# Check for missing values and handle them
print("\nMissing values per column:")
print(selected_data.isnull().sum())
selected_data = selected_data.dropna()  # Drop rows with missing values

# Features and labels
features = selected_data[feature_columns]
labels = selected_data[label_column]

# Encode labels if they are categorical
if labels.dtypes == 'object':
   labels = labels.astype('category').cat.codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

# Display the shape of training and testing data
print("\nShape of training data:", X_train.shape)
print("Shape of testing data:", X_test.shape)

# Train a KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print("\nModel accuracy on the test set: {:.2f}%".format(accuracy * 100))

# Prediction for a new sample (example feature values based on dataset range)
new_sample = np.array([[300000, 1200000, 280000, 1150000]])  # Replace with actual feature values
prediction = knn.predict(new_sample)
print("\nPrediction for the new sample:", prediction)

# Optional: Visualize feature relationships (scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(features[feature_columns[0]], features[feature_columns[1]], c=labels, cmap='viridis')
plt.colorbar(label='Label (Division)')
plt.xlabel(feature_columns[0])
plt.ylabel(feature_columns[1])
plt.title('Feature Scatter Plot')
plt.show()


# In[17]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load the dataset
data_path = r'C:\Users\Muttaki\Downloads\bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)

# Preview the dataset
print("Dataset preview:")
print(data.head())

# Selecting features and label based on the dataset structure
feature_columns = [
    'Household_Total',
    'Population_Total',
    'Household_Excluding_Slum_Floating',
    'Population_Excluding_Slum_Floating'
]  # Features
label_column = 'Division'  # Label

# Filter the dataset to include only the selected features and label
selected_data = data[feature_columns + [label_column]]

# Check for missing values and handle them
print("\nMissing values per column:")
print(selected_data.isnull().sum())
selected_data = selected_data.dropna()  # Drop rows with missing values

# Features and labels
features = selected_data[feature_columns]
labels = selected_data[label_column]

# Encode labels if they are categorical
if labels.dtypes == 'object':
    labels = labels.astype('category').cat.codes

# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.25, random_state=0)

# Train a KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print("\nModel accuracy on the test set: {:.2f}%".format(accuracy * 100))

# Prediction for a new sample (example feature values based on dataset range)
new_sample = np.array([[300000, 1200000, 280000, 1150000]])  # Replace with actual feature values
scaled_sample = scaler.transform(new_sample)
prediction = knn.predict(scaled_sample)
print("\nPrediction for the new sample:", prediction)

# Visualization (optional)
plt.figure(figsize=(10, 6))
plt.scatter(features[feature_columns[0]], features[feature_columns[1]], c=labels, cmap='viridis')
plt.colorbar(label='Label (Division)')
plt.xlabel(feature_columns[0])
plt.ylabel(feature_columns[1])
plt.title('Feature Scatter Plot')
plt.show()


# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data_path = r'C:\Users\Muttaki\Downloads\bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)

# Preview the dataset
print("Dataset preview:")
print(data.head())

# Selecting features and label based on the dataset structure
feature_columns = [
    'Household_Total',
    'Population_Total',
    'Household_Excluding_Slum_Floating',
    'Population_Excluding_Slum_Floating'
]  # Features
label_column = 'Division'  # Label

# Filter the dataset to include only the selected features and label
selected_data = data[feature_columns + [label_column]]

# Check for missing values and handle them
print("\nMissing values per column:")
print(selected_data.isnull().sum())
selected_data = selected_data.dropna()  # Drop rows with missing values

# Features and labels
features = selected_data[feature_columns]
labels = selected_data[label_column]

# Encode labels if they are categorical
if labels.dtypes == 'object':
    labels = labels.astype('category').cat.codes

# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.25, random_state=0)

# Train a KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print("\nKNN Model accuracy on the test set: {:.2f}%".format(accuracy * 100))

# Prediction for a new sample (example feature values based on dataset range)
new_sample = np.array([[300000, 1200000, 280000, 1150000]])  # Replace with actual feature values
scaled_sample = scaler.transform(new_sample)
prediction = knn.predict(scaled_sample)
print("\nKNN Prediction for the new sample:", prediction)

# Linear Regression
print("\nLinear Regression:")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_predictions = lin_reg.predict(X_test)
print("Linear Regression Predictions (first 5):", lin_reg_predictions[:5])

# Polynomial Regression
print("\nPolynomial Regression:")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)
poly_predictions = poly_reg.predict(X_test_poly)
print("Polynomial Regression Predictions (first 5):", poly_predictions[:5])

# Logistic Regression (For binary classification tasks)
print("\nLogistic Regression:")
log_reg = LogisticRegression(max_iter=1000)
y_train_binary = (y_train == 0).astype(int)  # Example binary classification
log_reg.fit(X_train, y_train_binary)
log_predictions = log_reg.predict(X_test)
print("Logistic Regression Predictions (first 5):", log_predictions[:5])
log_accuracy = accuracy_score((y_test == 0).astype(int), log_predictions)
print("Logistic Regression Accuracy: {:.2f}%".format(log_accuracy * 100))

# Visualization (optional)
plt.figure(figsize=(10, 6))
plt.scatter(features[feature_columns[0]], features[feature_columns[1]], c=labels, cmap='viridis')
plt.colorbar(label='Label (Division)')
plt.xlabel(feature_columns[0])
plt.ylabel(feature_columns[1])
plt.title('Feature Scatter Plot')
plt.show()


# In[19]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the Bangladesh Census dataset
data_path =r'C:\Users\Muttaki\Downloads\bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)

# Preview the dataset
print("Dataset preview:")
print(data.head())

# Selecting features and label based on the dataset structure
feature_columns = [
    'Household_Total',
    'Population_Total',
    'Household_Excluding_Slum_Floating',
    'Population_Excluding_Slum_Floating'
]  # Features
label_column = 'Literacy Rate_5year+_Overall'  # Label for regression
classification_column = 'Urban_Literacy Rate_7year+_Overall'  # Label for classification

# Filter the dataset to include only the selected features and labels
selected_data = data[feature_columns + [label_column, classification_column]]

# Check for missing values and handle them
print("\nMissing values per column:")
print(selected_data.isnull().sum())
selected_data = selected_data.dropna()  # Drop rows with missing values

# Features and labels for regression and classification
features = selected_data[feature_columns]
regression_labels = selected_data[label_column]
classification_labels = (selected_data[classification_column] > 70).astype(int)  # Binary classification

# Encode labels if they are categorical
if regression_labels.dtypes == 'object':
    regression_labels = regression_labels.astype('category').cat.codes

# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets for regression and classification
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    scaled_features, regression_labels, test_size=0.25, random_state=0
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    scaled_features, classification_labels, test_size=0.25, random_state=0
)

# Linear Regression
print("\nLinear Regression:")
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
lin_reg_predictions = lin_reg.predict(X_test_reg)
lin_reg_mse = mean_squared_error(y_test_reg, lin_reg_predictions)
print("Linear Regression MSE:", lin_reg_mse)
print("Linear Regression Predictions (first 5):", lin_reg_predictions[:5])

# Polynomial Regression
print("\nPolynomial Regression:")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train_reg)
X_test_poly = poly.transform(X_test_reg)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train_reg)
poly_predictions = poly_reg.predict(X_test_poly)
poly_mse = mean_squared_error(y_test_reg, poly_predictions)
print("Polynomial Regression MSE:", poly_mse)
print("Polynomial Regression Predictions (first 5):", poly_predictions[:5])

# Logistic Regression
print("\nLogistic Regression:")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_clf, y_train_clf)
log_predictions = log_reg.predict(X_test_clf)
log_accuracy = accuracy_score(y_test_clf, log_predictions)
print("Logistic Regression Accuracy: {:.2f}%".format(log_accuracy * 100))
print("Logistic Regression Predictions (first 5):", log_predictions[:5])

# KNN Classifier (for comparison)
print("\nKNN Classifier:")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_clf, y_train_clf)
knn_predictions = knn.predict(X_test_clf)
knn_accuracy = accuracy_score(y_test_clf, knn_predictions)
print("KNN Accuracy: {:.2f}%".format(knn_accuracy * 100))
print("KNN Predictions (first 5):", knn_predictions[:5])

# Visualization (optional)
plt.figure(figsize=(10, 6))
plt.scatter(features[feature_columns[0]], features[feature_columns[1]], c=classification_labels, cmap='viridis')
plt.colorbar(label='Binary Classification (0/1)')
plt.xlabel(feature_columns[0])
plt.ylabel(feature_columns[1])
plt.title('Feature Scatter Plot for Classification')
plt.show()


# In[20]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the Bangladesh Census dataset
data_path = r'C:\Users\Muttaki\Downloads\bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)

# Preview the dataset
print("Dataset preview:")
print(data.head())

# Output the total number of rows and columns in the dataset
print("\nTotal number of rows and columns in the dataset:", data.shape)

# Selecting features and label based on the dataset structure
feature_columns = [
    'Household_Total',
    'Population_Total',
    'Household_Excluding_Slum_Floating',
    'Population_Excluding_Slum_Floating'
]  # Features
label_column = 'Literacy Rate_5year+_Overall'  # Label for regression
classification_column = 'Urban_Literacy Rate_7year+_Overall'  # Label for classification

# Filter the dataset to include only the selected features and labels
selected_data = data[feature_columns + [label_column, classification_column]]

# Check for missing values and handle them
print("\nMissing values per column:")
print(selected_data.isnull().sum())
selected_data = selected_data.dropna()  # Drop rows with missing values

# Features and labels for regression and classification
features = selected_data[feature_columns]
regression_labels = selected_data[label_column]
classification_labels = (selected_data[classification_column] > 70).astype(int)  # Binary classification

# Encode labels if they are categorical
if regression_labels.dtypes == 'object':
    regression_labels = regression_labels.astype('category').cat.codes

# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets for regression and classification
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    scaled_features, regression_labels, test_size=0.25, random_state=0
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    scaled_features, classification_labels, test_size=0.25, random_state=0
)

# Linear Regression
print("\nLinear Regression:")
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
lin_reg_predictions = lin_reg.predict(X_test_reg)
lin_reg_mse = mean_squared_error(y_test_reg, lin_reg_predictions)
print("Linear Regression MSE:", lin_reg_mse)
print("Linear Regression Predictions (first 5):", lin_reg_predictions[:5])

# Polynomial Regression
print("\nPolynomial Regression:")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train_reg)
X_test_poly = poly.transform(X_test_reg)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train_reg)
poly_predictions = poly_reg.predict(X_test_poly)
poly_mse = mean_squared_error(y_test_reg, poly_predictions)
print("Polynomial Regression MSE:", poly_mse)
print("Polynomial Regression Predictions (first 5):", poly_predictions[:5])

# Logistic Regression
print("\nLogistic Regression:")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_clf, y_train_clf)
log_predictions = log_reg.predict(X_test_clf)
log_accuracy = accuracy_score(y_test_clf, log_predictions)
print("Logistic Regression Accuracy: {:.2f}%".format(log_accuracy * 100))
print("Logistic Regression Predictions (first 5):", log_predictions[:5])

# KNN Classifier (for comparison)
print("\nKNN Classifier:")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_clf, y_train_clf)
knn_predictions = knn.predict(X_test_clf)
knn_accuracy = accuracy_score(y_test_clf, knn_predictions)
print("KNN Accuracy: {:.2f}%".format(knn_accuracy * 100))
print("KNN Predictions (first 5):", knn_predictions[:5])

# Visualization (optional)
plt.figure(figsize=(10, 6))
plt.scatter(features[feature_columns[0]], features[feature_columns[1]], c=classification_labels, cmap='viridis')
plt.colorbar(label='Binary Classification (0/1)')
plt.xlabel(feature_columns[0])
plt.ylabel(feature_columns[1])
plt.title('Feature Scatter Plot for Classification')
plt.show()


# In[21]:


#Naive Bayes 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the Bangladesh Census dataset
data_path = r'C:\Users\Muttaki\Downloads\bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)

# Preview the dataset
print("Dataset preview:")
print(data.head())

# Output the total number of rows and columns in the dataset
print("\nTotal number of rows and columns in the dataset:", data.shape)

# Selecting features and label based on the dataset structure
feature_columns = [
    'Household_Total',
    'Population_Total',
    'Household_Excluding_Slum_Floating',
    'Population_Excluding_Slum_Floating'
]  # Features
label_column = 'Literacy Rate_5year+_Overall'  # Label for regression
classification_column = 'Urban_Literacy Rate_7year+_Overall'  # Label for classification

# Filter the dataset to include only the selected features and labels
selected_data = data[feature_columns + [label_column, classification_column]]

# Check for missing values and handle them
print("\nMissing values per column:")
print(selected_data.isnull().sum())
selected_data = selected_data.dropna()  # Drop rows with missing values

# Features and labels for regression and classification
features = selected_data[feature_columns]
regression_labels = selected_data[label_column]
classification_labels = (selected_data[classification_column] > 70).astype(int)  # Binary classification

# Encode labels if they are categorical
if regression_labels.dtypes == 'object':
    regression_labels = regression_labels.astype('category').cat.codes

# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets for regression and classification
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    scaled_features, regression_labels, test_size=0.25, random_state=0
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    scaled_features, classification_labels, test_size=0.25, random_state=0
)

# Linear Regression
print("\nLinear Regression:")
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
lin_reg_predictions = lin_reg.predict(X_test_reg)
lin_reg_mse = mean_squared_error(y_test_reg, lin_reg_predictions)
print("Linear Regression MSE:", lin_reg_mse)
print("Linear Regression Predictions (first 5):", lin_reg_predictions[:5])

# Polynomial Regression
print("\nPolynomial Regression:")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train_reg)
X_test_poly = poly.transform(X_test_reg)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train_reg)
poly_predictions = poly_reg.predict(X_test_poly)
poly_mse = mean_squared_error(y_test_reg, poly_predictions)
print("Polynomial Regression MSE:", poly_mse)
print("Polynomial Regression Predictions (first 5):", poly_predictions[:5])

# Logistic Regression
print("\nLogistic Regression:")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_clf, y_train_clf)
log_predictions = log_reg.predict(X_test_clf)
log_accuracy = accuracy_score(y_test_clf, log_predictions)
print("Logistic Regression Accuracy: {:.2f}%".format(log_accuracy * 100))
print("Logistic Regression Predictions (first 5):", log_predictions[:5])

# KNN Classifier (for comparison)
print("\nKNN Classifier:")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_clf, y_train_clf)
knn_predictions = knn.predict(X_test_clf)
knn_accuracy = accuracy_score(y_test_clf, knn_predictions)
print("KNN Accuracy: {:.2f}%".format(knn_accuracy * 100))
print("KNN Predictions (first 5):", knn_predictions[:5])

# Naive Bayes Classifier
print("\nNaive Bayes Classifier:")
nb = GaussianNB()
nb.fit(X_train_clf, y_train_clf)
nb_predictions = nb.predict(X_test_clf)
nb_accuracy = accuracy_score(y_test_clf, nb_predictions)
print("Naive Bayes Accuracy: {:.2f}%".format(nb_accuracy * 100))
print("Naive Bayes Predictions (first 5):", nb_predictions[:5])

# Visualization (optional)
plt.figure(figsize=(10, 6))
plt.scatter(features[feature_columns[0]], features[feature_columns[1]], c=classification_labels, cmap='viridis')
plt.colorbar(label='Binary Classification (0/1)')
plt.xlabel(feature_columns[0])
plt.ylabel(feature_columns[1])
plt.title('Feature Scatter Plot for Classification')
plt.show()


# In[22]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the Bangladesh Census dataset //d
data_path =  r'C:\Users\Muttaki\Downloads\bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)
selected_data = selected_data.dropna()
# Preview the dataset//d
print("Dataset preview:")
print(data.head())

# Output the total number of rows and columns in the dataset//d
print("\nTotal number of rows and columns in the dataset:", data.shape)

# Selecting features and label based on the dataset structure//d
feature_columns = [
    'Household_Total',
    'Population_Total',
    'Household_Excluding_Slum_Floating',
    'Population_Excluding_Slum_Floating'
]  # Features
classification_column = 'Urban_Literacy Rate_7year+_Overall'  # Label for classification
regression_label_column = 'Population_Total'  # Label for regression//d

# Filter the dataset to include only the selected features and labels//d
selected_data = data[feature_columns + [classification_column, regression_label_column]]

# Check for missing values and handle them
print("\nMissing values per column:")//d
print(selected_data.isnull().sum())//d
selected_data = selected_data.dropna()  # Drop rows with missing values

# Features and labels for classification and regression
features = selected_data[feature_columns]
classification_labels = (selected_data[classification_column] > 70).astype(int)  # Binary classification
regression_labels = selected_data[regression_label_column]

# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets for classification and regression
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    scaled_features, classification_labels, test_size=0.25, random_state=0
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    scaled_features, regression_labels, test_size=0.25, random_state=0
)

# KNN Classifier
print("\nKNN Classifier:")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_clf, y_train_clf)
knn_predictions = knn.predict(X_test_clf)
knn_accuracy = accuracy_score(y_test_clf, knn_predictions)
print("KNN Accuracy: {:.2f}%".format(knn_accuracy * 100))
print("KNN Predictions (first 5):", knn_predictions[:5])

# Decision Tree Classifier
print("\nDecision Tree Classifier:")
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train_clf, y_train_clf)
dt_predictions = dt.predict(X_test_clf)
dt_accuracy = accuracy_score(y_test_clf, dt_predictions)
print("Decision Tree Accuracy: {:.2f}%".format(dt_accuracy * 100))
print("Decision Tree Predictions (first 5):", dt_predictions[:5])

# Naive Bayes Classifier
print("\nNaive Bayes Classifier:")
nb = GaussianNB()
nb.fit(X_train_clf, y_train_clf)
nb_predictions = nb.predict(X_test_clf)
nb_accuracy = accuracy_score(y_test_clf, nb_predictions)
print("Naive Bayes Accuracy: {:.2f}%".format(nb_accuracy * 100))
print("Naive Bayes Predictions (first 5):", nb_predictions[:5])

# Logistic Regression
print("\nLogistic Regression:")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_clf, y_train_clf)
log_reg_predictions = log_reg.predict(X_test_clf)
log_reg_accuracy = accuracy_score(y_test_clf, log_reg_predictions)
print("Logistic Regression Accuracy: {:.2f}%".format(log_reg_accuracy * 100))
print("Logistic Regression Predictions (first 5):", log_reg_predictions[:5])

# Linear Regression
print("\nLinear Regression:")
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
lin_reg_predictions = lin_reg.predict(X_test_reg)
lin_reg_mse = mean_squared_error(y_test_reg, lin_reg_predictions)
print("Linear Regression MSE: {:.2f}".format(lin_reg_mse))
print("Linear Regression Predictions (first 5):", lin_reg_predictions[:5])

# Visualization (optional)
plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=classification_labels, cmap='viridis')
plt.colorbar(label='Binary Classification (0/1)')
plt.xlabel(feature_columns[0])
plt.ylabel(feature_columns[1])
plt.title('Feature Scatter Plot for Classification')
plt.show()

# Neural Network for Classification
print("\nNeural Network Classifier:")
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X_train_clf.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # For binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_clf, y_train_clf, epochs=20, batch_size=10, verbose=1)

# Evaluate Neural Network
nn_loss, nn_accuracy = model.evaluate(X_test_clf, y_test_clf, verbose=0)
print("Neural Network Accuracy: {:.2f}%".format(nn_accuracy * 100))


# In[ ]:


#Correlation Analysis:
import seaborn as sns
correlation_matrix = selected_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[ ]:


#Imbalanced Dataset:
classification_labels.value_counts().plot(kind='bar', title="Class Distribution")


# In[10]:


#Dataset Pre-processing
selected_data = selected_data.dropna()


# In[11]:


# Feature Scaling
scaled_features[:5]  # Display first 5 rows of scaled data


# In[12]:


#Model Selection & Comparison Analysis
#Accuracy Comparison:
models = ['KNN', 'Decision Tree', 'Logistic Regression', 'Naive Bayes']
accuracies = [knn_accuracy, dt_accuracy, log_reg_accuracy, nb_accuracy]
plt.bar(models, accuracies, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.show()


# In[13]:


#Confusion Matrix:
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test_clf, knn_predictions)  # Replace knn_predictions with predictions from other models
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix for KNN")
plt.show()


# In[14]:


models = ['KNN', 'Decision Tree', 'Naive Bayes', 'Logistic Regression', 'Neural Network']
accuracies = [knn_accuracy, dt_accuracy, nb_accuracy, log_reg_accuracy, nn_accuracy]


# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the Bangladesh Census dataset
data_path = 'bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)

# Preview the dataset
print("Dataset preview:")
print(data.head())

# Output the total number of rows and columns in the dataset
print("\nTotal number of rows and columns in the dataset:", data.shape)

# Selecting features and label based on the dataset structure
feature_columns = [
    'Household_Total',
    'Population_Total',
    'Household_Excluding_Slum_Floating',
    'Population_Excluding_Slum_Floating'
]  # Features
classification_column = 'Urban_Literacy Rate_7year+_Overall'  # Label for classification
regression_label_column = 'Population_Total'  # Label for regression

# Filter the dataset to include only the selected features and labels
selected_data = data[feature_columns + [classification_column, regression_label_column]]

# Check for missing values and handle them
print("\nMissing values per column:")
print(selected_data.isnull().sum())
selected_data = selected_data.dropna()  # Drop rows with missing values

# Features and labels for classification and regression
features = selected_data[feature_columns]
classification_labels = (selected_data[classification_column] > 70).astype(int)  # Binary classification
regression_labels = selected_data[regression_label_column]

# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets for classification and regression
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    scaled_features, classification_labels, test_size=0.25, random_state=0
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    scaled_features, regression_labels, test_size=0.25, random_state=0
)

# KNN Classifier
print("\nKNN Classifier:")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_clf, y_train_clf)
knn_predictions = knn.predict(X_test_clf)
knn_accuracy = accuracy_score(y_test_clf, knn_predictions)
print("KNN Accuracy: {:.2f}%".format(knn_accuracy * 100))

# Decision Tree Classifier
print("\nDecision Tree Classifier:")
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train_clf, y_train_clf)
dt_predictions = dt.predict(X_test_clf)
dt_accuracy = accuracy_score(y_test_clf, dt_predictions)
print("Decision Tree Accuracy: {:.2f}%".format(dt_accuracy * 100))

# Naive Bayes Classifier
print("\nNaive Bayes Classifier:")
nb = GaussianNB()
nb.fit(X_train_clf, y_train_clf)
nb_predictions = nb.predict(X_test_clf)
nb_accuracy = accuracy_score(y_test_clf, nb_predictions)
print("Naive Bayes Accuracy: {:.2f}%".format(nb_accuracy * 100))

# Logistic Regression
print("\nLogistic Regression:")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_clf, y_train_clf)
log_reg_predictions = log_reg.predict(X_test_clf)
log_reg_accuracy = accuracy_score(y_test_clf, log_reg_predictions)
print("Logistic Regression Accuracy: {:.2f}%".format(log_reg_accuracy * 100))

# Neural Network for Classification
print("\nNeural Network Classifier:")
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X_train_clf.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # For binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_clf, y_train_clf, epochs=20, batch_size=10, verbose=0)

# Evaluate Neural Network
nn_loss, nn_accuracy = model.evaluate(X_test_clf, y_test_clf, verbose=0)
print("Neural Network Accuracy: {:.2f}%".format(nn_accuracy * 100))

# Model selection and comparison analysis
models = ['KNN', 'Decision Tree', 'Naive Bayes', 'Logistic Regression', 'Neural Network']
accuracies = [knn_accuracy, dt_accuracy, nb_accuracy, log_reg_accuracy, nn_accuracy]

# Bar chart showcasing accuracy of all models
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.xlabel('Models')
plt.ylim(0, 1)
plt.show()

# Precision, recall, F1-score comparison for classification models
print("\nPrecision, Recall, and F1-Score Comparison:")
for model_name, predictions in zip(models[:-1], [knn_predictions, dt_predictions, nb_predictions, log_reg_predictions]):
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_clf, predictions, average='binary')
    print(f"{model_name} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# Heatmaps for each model
print("\nHeatmaps for Confusion Matrices:")
for model_name, predictions in zip(models[:-1], [knn_predictions, dt_predictions, nb_predictions, log_reg_predictions]):
    cm = confusion_matrix(y_test_clf, predictions, labels=np.unique(y_test_clf))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlBu", xticklabels=np.unique(y_test_clf), yticklabels=np.unique(y_test_clf))
    plt.title(f"Heatmap for {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Heatmap for Neural Network
nn_predictions = (model.predict(X_test_clf) > 0.5).astype(int).flatten()
cm_nn = confusion_matrix(y_test_clf, nn_predictions, labels=np.unique(y_test_clf))
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="RdYlBu", xticklabels=np.unique(y_test_clf), yticklabels=np.unique(y_test_clf))
plt.title("Heatmap for Neural Network")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




