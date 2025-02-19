# Titanic - binary classification problem

## Purpose of the work
In my work I would like to compare a few simple classification models, which are designed to predict which passengers have survived. 

## Methods used for classification:
* Logistic regression
* KNN - K-Nearest Neighbors
* SVM - Support Vector Machines
* Decision Tree
* Random Forest
* Gradient Boosting

## EDA - Exploratory data analysis

Data contains 12 features. At the begining I deleted 4 of them: PassangerId, Name, Ticket and Cabin. First three features are not important in case of classification. 
Last variable has many missing values (96,5% off all passangers). I think this is because some passengers didn't have a cabin, but I decided to remove this information from the dataset anyway.

Looking at the correlation matrix the biggest impact on Survived variable has: Sex(0.54), Passanger class(-0.39) and Fare price paid(0.29).

| Column        | Description                                                                                           |
|---------------|-------------------------------------------------------------------------------------------------------|
| PassengerId   | Unique ID                                                                                             |
| Survived      | Survival status (0 = No, 1 = Yes)                                                                     |
| Pclass        | Passenger class (1 = First, 2 = Second, 3 = Third)                                                    |
| Name          | Name of passenger                                                                                     |
| Sex           | Gender of passenger                                                                                   |
| Age           | Age of passenger                                                                                      |
| SibSp         | Number of siblings and spouses aboard the Titanic                                                     |
| Parch         | Number of children and parents aboard the Titanic                                                     |
| Ticket        | Unique ticket ID                                                                                      |
| Fare          | Fare price paid (in pounds)                                                                           |
| Cabin         | Passenger's cabin number                                                                              |
| Embarked      | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)                                  |

The data was split into a training and test set in a 7:3 ratio.
The data were on different scales, so I used StandardScaler to change them so that the mean was 0 and the standard deviation was 1.

## Choosing parameters

For KNN, SVM and Random Forest I use GridSearchCV for searching the best results. 
When working with decision trees, I concentrated on finding the optimal value of maximum depth for each tree. The plot below makes it clear that the best value is 4, because for bigger values model is overfitting.
![image](https://github.com/user-attachments/assets/becbe915-894a-4953-9ceb-4330db1ada00)

## Comparison of results

For the models comparison I choose ROC-AUC measure and F1-score.

![image](https://github.com/user-attachments/assets/53f6fae9-8b27-485b-8fa1-32b14812c4cf)

Looking only on the graph it is hard to say which model is the best. Only the exact AUC scores show which models are better and which are worse.

| Method             | AUC      |         | Method            | F1 Score  |
|--------------------|----------|---------|-------------------|-----------|
| KNN                | 0.846230 |         | Gradient Boost    | 0.758621  |
| Gradient Boosting  | 0.845642 |         | KNN               | 0.748538  |
| Linear Regression  | 0.844058 |         | Decision Tree     | 0.743590  |
| Random Forest      | 0.836863 |         | SVM               | 0.737500  |
| Decision Tree      | 0.819395 |         | Random Forest     | 0.737500  |
| SVM                | 0.816273 |         | Linear Regression | 0.728324  |

In summary, the best methods are K-nearest neighbours and Gradient boost, beacause both methods have the best AUC and F1 scores.

