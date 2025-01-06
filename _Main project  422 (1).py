#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import seaborn as sns
data_path =  r'C:\Users\Muttaki\Downloads\bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)
#selected_data = selected_data.dropna()
#selected_data = selected_data.dropna()
print("=====Dataset preview=====")
print(data.head())
selected_data = data[feature_columns + [classification_column, regression_label_column]] #filtering 
print("=====Selected Data===== ")
print(selected_data)
print("===Miss_values===")
print(selected_data.isnull().sum())


# In[3]:


print("\nT_num of rows and columns in there ", data.shape)


# In[5]:


import seaborn as sns
#correlation_matrix = selected_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
data_path =r'C:\Users\Muttaki\Downloads\bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx'
data = pd.read_excel(data_path)
classification_column = 'Urban_Literacy Rate_7year+_Overall'  
classification_labels = (data[classification_column] > 70).astype(int)   
print("Imbalanced Dataset:")
class_counts = classification_labels.value_counts()
print("Class Distribution:")
print(class_counts)
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title("Class Distribution for Output Feature")
plt.xlabel("Classes")
plt.ylabel("Number of Instances")
plt.xticks([0, 1], ['Class 0', 'Class 1'], rotation=0)
plt.show()


# In[9]:


feature_columns = [
    'Household_Total',
    'Population_Total',
    'Household_Excluding_Slum_Floating',
    'Population_Excluding_Slum_Floating']
classification_column = 'Urban_Literacy Rate_7year+_Overall'   #lr
regression_label_column = 'Population_Total'   #lr
features = selected_data[feature_columns]
classification_labels = (selected_data[classification_column] > 70).astype(int)  # 0.1classify
regression_labels = selected_data[regression_label_column]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features) # f_sc
print('--------feature_columns------')
print(feature_columns)
print("------features-------")
print(features)
print("-----classification_column-------")
print(classification_column)
print("------regression_label_column-------")
print(regression_label_column)
print("-------classification-labels---------")
print(classification_labels)
print('------regression_labels------')
print(regression_labels)
print("--------scaler-------")
print(scaler)
print("---------scaled_features-----")
print(scaled_features)


# In[10]:


#data split data train_taste
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(scaled_features, classification_labels, test_size=0.25, random_state=0)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(scaled_features, regression_labels, test_size=0.25, random_state=0)
print("-----X_train_clf, X_test_clf, y_train_clf, y_test_clf-------")
print(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
print("------X_train_reg, X_test_reg, y_train_reg, y_test_reg -------")
print(X_train_reg, X_test_reg, y_train_reg, y_test_reg )


# In[11]:


#KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_clf, y_train_clf)
knn_predictions = knn.predict(X_test_clf)
knn_accuracy = accuracy_score(y_test_clf, knn_predictions)
print("KNN Accuracy: {:.2f}%".format(knn_accuracy * 100))
print("KNN_predictions:")
print(knn_predictions)
print("-----knn_accuracy -----")
print(knn_accuracy ) #(-1-1)

#print("\nClassification  for KNN:")
#print(classification_report(y_test_clf, knn_predictions))


# In[12]:


print("Decision_tree :")
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train_clf, y_train_clf)
dt_predictions = dt.predict(X_test_clf)
dt_accuracy = accuracy_score(y_test_clf, dt_predictions)
print("Decision Tree Accuracy: {:.2f}%".format(dt_accuracy * 100))
print("----dt_predictions-----")
print(dt_predictions)
print("-------dt_accuracy------")
print(dt_accuracy)#(-1-1)


# In[13]:


print("Logistic Regression:")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_clf, y_train_clf)
log_reg_predictions = log_reg.predict(X_test_clf)
log_reg_accuracy = accuracy_score(y_test_clf, log_reg_predictions)
print("Logistic Regression Accuracy: {:.2f}%".format(log_reg_accuracy * 100))
print("-----log_reg_predictions -----")
print(log_reg_predictions )
print("------log_reg_accuracy----")
print(log_reg_accuracy)


# In[14]:


print("Naive Bayes :")
nb = GaussianNB()
nb.fit(X_train_clf, y_train_clf)
nb_predictions = nb.predict(X_test_clf)
nb_accuracy = accuracy_score(y_test_clf, nb_predictions)
print("Naive Bayes Accuracy: {:.2f}%".format(nb_accuracy * 100))
print('--------nb_predictions-------')
print(nb_predictions)
print("-----nb_accuracy-----")
print(nb_accuracy)


# In[15]:


models = ['KNN', 'Decision Tree', 'Naive Bayes']
accuracies = [knn_accuracy, dt_accuracy, nb_accuracy]


# In[16]:


#Bar-chart
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color='blue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.xlabel('Models')
plt.ylim(0, 1)
plt.show()


# In[17]:


#precition recall
from sklearn.metrics import precision_recall_fscore_support

for model_name, predictions in zip(models[:-1], [knn_predictions, dt_predictions, nb_predictions, log_reg_predictions]):
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_clf, predictions, average='binary')
    print(f"{model_name} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    print(len(y_test_clf), len(predictions))


# In[18]:


# Confusion Matrix for each model
for model_name, predictions in zip(models[:-1], [knn_predictions, dt_predictions, nb_predictions, log_reg_predictions]):
    cm = confusion_matrix(y_test_clf, predictions, labels=np.unique(y_test_clf))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_clf))
    disp.plot(cmap='RdYlBu')  #rgb
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()


# In[19]:


#heatmap 
for model_name, predictions in zip(models[:-1], [knn_predictions, dt_predictions, nb_predictions, log_reg_predictions]):
    cm = confusion_matrix(y_test_clf, predictions, labels=np.unique(y_test_clf))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlBu", xticklabels=np.unique(y_test_clf), yticklabels=np.unique(y_test_clf))
    plt.title(f"Heatmap for {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




