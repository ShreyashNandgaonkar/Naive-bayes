#Importing libraries
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
#Dataset
dataset = datasets.load_iris()
#Scikit learn
model = GaussianNB()
model.fit(dataset.data, dataset.target)
#model prediction
expected = dataset.target
predicted = model.predict(dataset.data)
#accuracy and confusion matrix
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))