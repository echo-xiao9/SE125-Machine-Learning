from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import csv
import time


X_train_file = csv.reader(open('../data/X_train.csv'))
Y_train_file = csv.reader(open('../data/Y_train.csv'))
X_test_file = csv.reader(open('../data/X_test.csv'))
Y_test_file = csv.reader(open('../data/Y_test.csv'))
X_train = []
Y_train = []
X_test = []
Y_test = []


def loadData():
    i = 0
    for content in X_train_file:
        i += 1
        if (i == 1):
            continue
        content = list(map(int, content))
        if len(content) != 0:
            X_train.append(content)
    i = 0;
    for content in Y_train_file:
        i += 1
        if (i == 1):
            continue
        content = list(map(int, content))
        if len(content) != 0:
            Y_train.append(content)
    i = 0;
    for content in X_test_file:
        i += 1
        if (i == 1):
            continue
        content = list(map(int, content))
        if len(content) != 0:
            X_test.append(content)

    i = 0;
    for content in Y_test_file:
        i += 1
        if (i == 1):
            continue
        content = list(map(int, content))
        if len(content) != 0:
            Y_test.append(content)


def bernNB():
    BernNB = BernoulliNB(binarize=True)
    BernNB.fit(X_train, Y_train)
    print(BernNB)
    y_expect = Y_test
    y_pred = BernNB.predict(X_test)
    print(accuracy_score(y_expect, y_pred))

def MultiNB():
    MultiNB = MultinomialNB()
    MultiNB.fit(X_train, Y_train)
    print(MultiNB)
    y_expect = Y_test
    y_pred = MultiNB.predict(X_test)
    print(accuracy_score(y_expect, y_pred))


def GausNB():
    GausNB = GaussianNB()
    GausNB.fit(X_train, Y_train)
    print(GausNB)
    y_expect = Y_test
    y_pred = GausNB.predict(X_test)
    print(accuracy_score(y_expect, y_pred))


def logisticRegrression():
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)
    y_expect = Y_test
    y_pred = log_reg.predict(X_test)
    print(accuracy_score(y_expect, y_pred))


startTime = time.time()
loadData()
endLoadTime = time.time()
logisticRegrression()
endTime =time.time()
print("load file time: %.2f\ncalculating time: %.2f \ntotal time: %.2f" % (endLoadTime-startTime, endTime-endLoadTime, endTime-startTime))