from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB


iris = load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size = 0.3)

gaussnb = GaussianNB()
gaussnb.fit(X_train, y_train)
y_pred = gaussnb.predict(X_test)
print("Gaussian Naive Bayes model accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))


categoricalnb = CategoricalNB()
categoricalnb.fit(X_train, y_train)
y_pred = categoricalnb.predict(X_test)
print("Categorical Naive Bayes model accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))