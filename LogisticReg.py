import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from DataReader import DataReader

data_reader = DataReader("./data/us_trial.text", "./data/us_trial.labels")
X, Y = data_reader.get_features()

len_X = len(X)
len_X = (int)(len_X / 10) 
len_train = (int)(0.8*len_X)
print(len_train)
trainX, trainY = X[0:len_train], Y[0:len_train]
trainX, trainY = np.asarray(trainX), np.asarray(trainY)

testX, testY = X[len_train:len_X], Y[len_train:len_X]
testX, testY = np.asarray(testX), np.asarray(testY)
print(len(testX))


if __name__ == '__main__':
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(trainX, trainY)

    counter, len_testX = 0, len(testX)
    for i in range(len_testX):
        output = clf.predict(np.asarray([testX[i]]))
        if output == testY[i]:
            counter += 1
    accuracy = counter / len_testX
    accuracy = accuracy * 100
    print("Accuracy: ", accuracy)