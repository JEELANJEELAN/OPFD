import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('Online_fraud.csv')
data.head()
data.info()
data.describe()
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))
int_ = (data.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))
fl = (data.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))
sns.countplot(x='type', data=data)
sns.barplot(x='type', y='amount', data=data)
data['isFraud'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame containing transaction data
sns.displot(data['step'], bins=50, kde=True)
plt.xlabel('step')
plt.ylabel('Density')
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame containing transaction data
numeric_data = data.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(12, 6))
sns.heatmap(numeric_data.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()
type_new = pd.get_dummies(data['type'], drop_first=True)
data_new = pd.concat([data, type_new], axis=1)
data_new.head()
X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = data_new['isFraud']
X.shape, y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=42)
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Assuming X and y are your original data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

models = [
    LogisticRegression(random_state=42),
    XGBClassifier(random_state=42),
    SVC(kernel='rbf', probability=True, random_state=42),
    RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=42)
]

for i, model in enumerate(models):
    model.fit(X_train, y_train)  # Proper model fitting with training data

    train_preds = model.predict_proba(X_train)[:, 1]
    print(f'{model.__class__.__name__} : ')

    # Use ROC AUC score for better evaluation for probability-based models
    train_auc = roc_auc_score(y_train, train_preds)
    print('Training ROC AUC: ', train_auc)

    test_preds = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    print('Validation ROC AUC: ', test_auc)
    print()
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix = ConfusionMatrixDisplay.from_estimator(models[1], X_test, y_test)
confusion_matrix.plot()
plt.show()
