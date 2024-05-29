# Heart Disease Prediction and Analysis

This project focuses on the analysis and prediction of heart diseases using various machine learning techniques. It includes exploratory data analysis (EDA) and the implementation of a voting classifier model combining Logistic Regression, K-Nearest Neighbors, and Random Forest.

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to have Python installed along with several libraries.

Ensure that you have the ucimlrepo package installed, as it is required for this project:
pip install ucimlrepo

**Data**
The dataset used in this project is from the UCI Machine Learning Repository. It includes various features related to heart health indicators.

**Exploratory Data Analysis (EDA)**
EDA is performed to understand the distribution and relationships between different features in the dataset. Visualizations and statistical summaries are included to provide insights into the data.

**Modeling**
**Initialize Individual Models**
We initialize individual models with different solvers for Logistic Regression:

clf1 = LogisticRegression(max_iter=200, solver='saga', random_state=42)
clf2 = KNeighborsClassifier()
clf3 = RandomForestClassifier(random_state=42)
**Combine Models into a Voting Classifier**
A voting classifier is used to combine the predictions of the individual models:

eclf = VotingClassifier(estimators=[
    ('lr', clf1), ('knn', clf2), ('rf', clf3)], voting='soft')
**Train the Voting Classifier**
The voting classifier is trained on the training dataset:

eclf.fit(X_train, y_train.ravel())

**Predict on the Test Set**
Predictions are made on the test set:

y_pred = eclf.predict(X_test)

Evaluate the Model
The model's performance is evaluated using accuracy, classification report, and confusion matrix:

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

**Evaluation**
The model's accuracy, classification report, and confusion matrix are provided to assess its performance.
**Results**
The results section includes the findings from the EDA and the performance metrics of the machine learning model.

**Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

