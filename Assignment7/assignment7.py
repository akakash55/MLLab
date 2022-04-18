from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from matplotlib import pyplot as plt

iris = load_iris()
# print(iris)
print(f"Output features: {iris['target_names']}")
arr = iris['target_names']
X=iris.data
y=iris.target
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4)


entropyclf = DecisionTreeClassifier(criterion="entropy")
entropyclf.fit(X_train,y_train)
y_pred=entropyclf.predict(X_test)
print("Accuracy of Test Data = ",accuracy_score(y_true = y_test, y_pred=y_pred))
print(confusion_matrix(y_test,y_pred))
fig, ax = plt.subplots(figsize=(6, 6))
tree.plot_tree(entropyclf,ax=ax)
plt.show()


giniclf = DecisionTreeClassifier(criterion="gini")
giniclf.fit(X_train,y_train)
y_pred=giniclf.predict(X_test)
print("Accuracy of Test Data = ",accuracy_score(y_true = y_test, y_pred=y_pred))
print(confusion_matrix(y_test,y_pred))
fig, ax = plt.subplots(figsize=(6, 6))
tree.plot_tree(giniclf,ax=ax)
plt.show()


def predictionFunction(featureValue):
  print(arr[featureValue])

for i in range(0, len(y_pred)):    
    predictionFunction(y_pred[i]),  