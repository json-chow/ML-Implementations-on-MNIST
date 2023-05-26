import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from multiclass import OneVsRestClassifier
from logisticregression import LogisticRegression
from knn import KNeighborsClassifier
from naivebayes import GaussianNB

data = sklearn.datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(data["data"], data["target"])
X_train, X_test = X_train.T, X_test.T

# Logistic Regression: ~90% acc, ~90% weighted f1
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train).clfs
print("Logistic Regression\n", classification_report(y_test, clf.predict(X_test)))

# K Neighbors (K=3): ~98% acc, ~98% weighted f1
clf = KNeighborsClassifier(3)
clf.fit(X_train, y_train)
print("K Nearest Neighbors\n", classification_report(y_test, clf.predict(X_test)))

# Gaussian Naive Bayes: ~75% acc, ~75% weighted f1
clf = GaussianNB()
clf.fit(X_train, y_train)
print("Gaussian Naive Bayes\n", classification_report(y_test, clf.predict(X_test)))