import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier



# read a date
link_to_data = "../.venv/train.csv"
dataset = pd.read_csv(link_to_data)
print(dataset.columns)
# PassengerId      - unique id
# Survived         - 0=no, 1=yes
# Pclass           - 1=first class, 2=second class, 3=third class
# Name             - name of passenge
# Sex              - gender
# Age              - age of passenger
# SibSp            - number of sibings an spouses of the passenger aboard the Titanic
# Parch            - number of children and parents of the passenger aboard the Titanic
# Ticket           - unique ticked id
# Fare             - paid price (in pounds)
# Cabin            - passenger's cabin number
# Embarked         - where passenger emarked, C = Cherbourg Q = Queenstown S = Southampton

# counting rows, which do not have a value
print(dataset.isna().sum())
# Output:
# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age             177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2

# Conclusion:
# In 3 of 12 columns there are missing values: Age - 177 values, Cabin - 687 columns, Embarked - 2
# The most missing values has Cabin value
print(dataset["Cabin"].unique())
# At the begining I delete this column from datset
# In case of Age and Fare column I delete only rows with missing values
dataset.drop(columns = "Cabin", inplace=True)
dataset.dropna(axis=0, inplace=True)
# Now dataset contains 11 columns with 331 rows

# Checking if categorical column are correct
dataset.nunique()
# Output:
# PassengerId    712
# Survived         2
# Pclass           3
# Name           712
# Sex              2
# Age             88
# SibSp            6
# Parch            7
# Ticket         541
# Fare           219
# Embarked         3
# dtype: int64

# Changing categorical columns using One-hot encoding
# Use copy to be sure that data in original table will not be modify during working with X and y tables
y = dataset['Survived'].copy()
X= dataset[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
dummies_s = pd.get_dummies(dataset['Sex'])
dummies_e = pd.get_dummies(dataset['Embarked'])
X = pd.concat([X, dummies_e, dummies_s], axis=1)
X = X.astype(int)

# ----------------------------------------------------------------------------------------------------------------------
#Spliting the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, stratify=y, random_state=42)

# Scaling the data that evey feature have mean 0 and standard deviation 1
# Scaling after spliting the data prevents information leakage into model
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------------------------------------------------------------------
# Logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
# Prediction
y_predict = lr.predict(X_test)

# Results
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))

# function to calculate probability from: https://www.w3schools.com/Python/python_ml_logistic_regression.asp
def logit2prob(logr, X):
  log_odds = logr.coef_ * X + logr.intercept_
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

# print(logit2prob(lr, X_test))

# Calculate correlation to see which featurest have the bigest weight
X.corr()
# Column survived has the highest correlation with variables female(0.536762), male(-0.536762) and Pclass(-0.356462).
# Now try to build model only on 2 the most important features(the male features skip, because is oposit the female features)
X2 = X[['female', 'Pclass']].copy()
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size =0.3, stratify=y, random_state=42)

X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.transform(X_test2)

lr.fit(X_train2, y_train2)
# Prediction
y_predict2 = lr.predict(X_test2)

# Results
print(classification_report(y_test2, y_predict2))
print(confusion_matrix(y_test2, y_predict2))

# Conclusion: Model builded on this 2 features has only 1% lower accuracy and it is euqal 77%

# ----------------------------------------------------------------------------------------------------------------------
# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

# testing diffrent k values, code from: https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/
neighbors = [1, 3, 5, 7, 9]
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# for i, k in enumerate(neighbors):
#   knn = KNeighborsClassifier(n_neighbors=k)
#   knn.fit(X_train, y_train)
#
#   train_accuracy[i] = knn.score(X_train, y_train)
#   test_accuracy[i] = knn.score(X_test, y_test)
#   print("Result for: ", k, "-nearest neighbours")
#   print(train_accuracy[i], test_accuracy[i])
#   y_predict_knn = knn.predict(X_test)
#   print(confusion_matrix(y_test, y_predict_knn))
#
# plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
# plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
#
# plt.legend()
# plt.xlabel('n_neighbors')
# plt.ylabel('Accuracy')
# plt.show()

# Conclusion: The best is model with 3 neighbors
# Accurancy: 81%

# # Testing K-nearest neighbour with 2 features
# gs = gridspec.GridSpec(2, 2)
# fig = plt.figure(figsize=(10,8))
#
# # Creating a 2x2 grid
# fig, axd = plt.subplot_mosaic(
#     [['1', '3'],
#      ['5', '7']],
#     figsize=(10, 8),
#     layout="constrained"
# )

# for i, (k, (name, ax)) in enumerate(zip([1, 3, 5, 7], axd.items())):
#   knn = KNeighborsClassifier(n_neighbors=k)
#   knn.fit(X_train2, y_train2)
#   y_train_nparray = y_train2.to_numpy()
#
#   plot_decision_regions(X=X_train2, y=y_train_nparray, ax=ax, clf=knn, legend=2)
#   ax.set_title(f'{k} nearest neighbors')
#   ax.set_xlabel('female')
#   ax.set_ylabel('class')
#
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# SVM
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_predict_svm = svm.predict(X_test)
print("Results for SVM method:")
print(svm.score(X_train, y_train), svm.score(X_test, y_test))
print(confusion_matrix(y_test, y_predict_svm))
# Output: Accurancy for training set is euqal 85% and for test set- 81%.
# What is quite good score, but try now to optimization parameters

parameters={
    'C':[0.3, 0.5, 1, 5, 10, 20],
    'gamma':[1, 0.1, 0.001]
}
best_param = GridSearchCV(svm, parameters)
best_param.fit(X_train, y_train)
print(best_param.best_params_)
# Result: {'C': 5, 'gamma': 0.1}
# I woudl like to test more value for C around 5
parameters2={
    'C':[3, 4, 5, 6, 7],
    'gamma': [0.1]
}
best_param2 = GridSearchCV(svm, parameters)
best_param2.fit(X_train, y_train)
print(best_param2.best_params_)
# There is any different and still the best value for C is 5 and for gamma is 0.1

# ----------------------------------------------------------------------------------------------------------------------
# Decision tree
tree_clf = tree.DecisionTreeClassifier(random_state=42)
tree_clf = tree_clf.fit(X_train, y_train)
y_predict_tree = tree_clf.predict(X_test)
print("Output for decision tree:")
print(tree_clf.score(X_train, y_train), tree_clf.score(X_test, y_test))
print(confusion_matrix(y_test, y_predict_tree))
# Output: Accurancy for training set is euqal 98% and for test set- 73%.
# This mean that model is overfitted
tree.plot_tree(tree_clf)
plt.show()
# Looking at the graph is understable, because it was created too many leafs
# Try to optimize model
best_tree = [0, 0];
for i in [2, 3, 4, 5, 6, 7, 8]:
  tree_clf = tree.DecisionTreeClassifier(random_state=42, max_depth=i)
  tree_clf = tree_clf.fit(X_train, y_train)
  y_predict_tree = tree_clf.predict(X_test)
  test_accurancy = tree_clf.score(X_test, y_test)
  if test_accurancy > best_tree[0]:
    best_tree[0] = test_accurancy
    best_tree[1] = i
  print(f"Output for decision tree that has max_depth = {i}:")
  print(tree_clf.score(X_train, y_train), test_accurancy)
  print(confusion_matrix(y_test, y_predict_tree))
print(f"The highest accurancy for testing set is for model with max_depth = {best_tree[1]} and it is euqal: {best_tree[0]})")

# Conclusion: The highest accurancy for testing dataset is when max_depth parameter is euqal 4 and it is euqal 81%
# Confusion matrix for this tree look like this:
# [[116  11]
#  [ 29  58]]

# Decision tree (2 features)
# Try PCA to reduce number of features
tree_clf_pca = tree.DecisionTreeClassifier(random_state=42)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

tree_clf_pca = tree_clf_pca.fit(X_train_pca, y_train)
y_predict_tree_pca = tree_clf_pca.predict(X_test_pca)

print("Output for decision tree using pca:")
print(tree_clf_pca.score(X_train_pca, y_train), tree_clf_pca.score(X_test_pca, y_test))
print(confusion_matrix(y_test, y_predict_tree_pca))
# Output: Accurancy for training set is euqal 98% and for test set- 48%.
# This mean that model is overfitted

tree.plot_tree(tree_clf_pca)
plt.show()
# Looking at the graph is understable, because it was created too many leafs

# Try to optimize model
best_tree_pca = [0, 0];
for i in [2, 3, 4, 5, 6]:
  tree_clf_pca = tree.DecisionTreeClassifier(random_state=42, max_depth=i)
  tree_clf_pca = tree_clf_pca.fit(X_train_pca, y_train)
  y_predict_tree_pca = tree_clf_pca.predict(X_test_pca)
  test_accurancy_pca = tree_clf_pca.score(X_test_pca, y_test)
  if test_accurancy_pca > best_tree_pca[0]:
    best_tree_pca[0] = test_accurancy_pca
    best_tree_pca[1] = i
  print(f"Output for decision tree that has max_depth = {i}:")
  print(tree_clf_pca.score(X_train_pca, y_train), test_accurancy_pca)
  print(confusion_matrix(y_test, y_predict_tree_pca))
print(f"The highest accurancy for testing set is for model with max_depth = "
      f"{best_tree_pca[1]} and it is euqal: {best_tree_pca[0]})")

# Conclusion: The highest accurancy for testing dataset with max_depth parameter=2 and it is euqal 66%
# Confusion matrix for this tree look like this:
# [[105  22]
#  [ 50  37]]
# This is worst score in comparison to model without pca

X_tree = dataset[['Pclass', 'Sex']].copy()
print(X_tree)
X_tree['Sex'] = X_tree['Sex'].astype('category')
X_tree['Sex'] = X_tree['Sex'].cat.codes
print(X_tree)
X_train_tree2, X_test_tree2, y_train_tree2, y_test_tree2 = train_test_split(X_tree, y, stratify=y,
                                                                            test_size=0.3,random_state=42)
tree2_clf = tree.DecisionTreeClassifier(random_state=42)
tree2_clf = tree2_clf.fit(X_train_tree2, y_train_tree2)
y_predict_tree2 = tree2_clf.predict(X_test_tree2)
print("Output for decision tree with 2 featrures class and sex:")
print(tree2_clf.score(X_train_tree2, y_train_tree2), tree2_clf.score(X_test_tree2, y_test_tree2))
print(confusion_matrix(y_test_tree2, y_predict_tree2))
# Output: Accurancy for training set is euqal 80% and for test set- 77%.
# Confusion matrix:
# [[124   3]
#  [ 47  40]]
tree.plot_tree(tree2_clf)
plt.show()

# The output tree has depth=3. I try set max_depth for 2 and the output does not change.

# ----------------------------------------------------------------------------------------------------------------------
# Random forest
random_forest_clf = RandomForestClassifier(max_depth=3, random_state=42)
random_forest_clf.fit(X_train, y_train)
y_predict_random_forest = random_forest_clf.predict(X_test)
print("Output for random forest:")
print(random_forest_clf.score(X_train, y_train), random_forest_clf.score(X_test, y_test))
print(confusion_matrix(y_test, y_predict_random_forest))
# Output: Accurancy for training set is euqal 82% and for test set- 79%.
# Confusion matrix:
# [[114  13]
#  [ 31  56]]

# Now it is quite good score, but try optimize parameters(max_depth, )
rf_params = {'n_estimators': [50, 100],
              'max_depth': [2, 3, 4, 5, 6]}
best_params_rf = GridSearchCV(random_forest_clf, rf_params)
best_params_rf.fit(X_train, y_train)
print("The best parameters found by GridSearchCV for random forest:")
print(best_params_rf.best_params_)
best_random_forest_clf = RandomForestClassifier(max_depth=5, random_state=42)
# n_estimator is set 100 by default
best_random_forest_clf.fit(X_train, y_train)
y_predict_best_random_forest = best_random_forest_clf.predict(X_test)
print(best_random_forest_clf.score(X_train, y_train), best_random_forest_clf.score(X_test, y_test))
# Output:
# The best parameters found by GridSearchCV for random forest:
# {'max_depth': 5, 'n_estimators': 100}
# Then accurancy for training set = 87% and for test set = 80%

# ----------------------------------------------------------------------------------------------------------------------
# Gradient boosting classifier
gbc = GradientBoostingClassifier(max_depth=3, random_state=42)
gbc.fit(X_train, y_train)
y_predict_gbc = gbc.predict(X_test)
print("Gradiend Boosting CLassifier")
print("Accurancy for training set and test set:")
print(gbc.score(X_train, y_train), gbc.score(X_test, y_test))
# Output:
# 0.9257028112449799 0.794392523364486
# Model probably is overfitted
gbc1 = GradientBoostingClassifier(max_depth=1, random_state=42)
gbc1.fit(X_train, y_train)
y_predict_gbc1 = gbc1.predict(X_test)
print("Gradiend Boosting CLassifier")
print("Accurancy for training set and test set:")
print(gbc1.score(X_train, y_train), gbc1.score(X_test, y_test))
# Output:
# 0.8072289156626506 0.8037383177570093
# This is definitely better result

# ----------------------------------------------------------------------------------------------------------------------
#