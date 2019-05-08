import numpy as np
import pandas as pd
import time
import random

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize

from sklearn import neighbors
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier




class SupervisedLearning:

    def __init__(self, dataset):
        self.dataset = dataset
        self.dtTrainAccuracy = []
        self.dtTestAccuracy = []
        self.dtCrossAccuracy = []
        self.dtTrainTime = []

        self.boostTrainAccuracy = []
        self.boostCrossAccuracy = []
        self.boostTestAccuracy = []
        self.boostTrainTime = []

        self.boostUnprunedCrossAccuracy = []

        self.knnTrainAccuracy = []
        self.knnTestAccuracy = []
        self.knnCrossAccuracy = []
        self.knnTrainTime = []
        self.knnTestTime_Train = []
        self.knnTestTime_Test = []


        self.svmTrainTime = {'linear': [], 'poly': [], 'rbf': []}
        self.svmTrainAccuracy = {'linear': [], 'poly': [], 'rbf': []}
        self.svmCrossAccuracy = {'linear': [], 'poly': [], 'rbf': []}
        self.svmTestAccuracy = {'linear': [], 'poly': [], 'rbf': []}


        self.mlpcTrainAccuracy = []
        self.mlpcCrossAccuracy = []
        self.mlpcTestAccuracy = []
        self.mlpcTrainTime = []
        pass

    def runExperiment(self):
        start = time.time()
        self.loadData(self.dataset)
        # self.expDecisionTree()
        # self.expBoosting()
        # self.expKnn()
        # self.expSvm()
        # self.expMlpc()
        end = time.time()
        print("Time Elapsed = %s seconds" %(str(end-start)))

    def loadData(self, dataset="forest"):
        if(dataset == "ford"):
            f = open("../data/ford/fordTrain.csv")
            trainData = pd.read_csv(f)
            trainData = trainData.drop(columns=["TrialID", "ObsNum"])
            f.close()

            f = open("../data/ford/fordTest.csv")
            testData = pd.read_csv(f)
            testData = testData.drop(columns=["TrialID", "ObsNum", "IsAlert"])
            f.close()


            f = open("../data/ford/fordSolution.csv")
            solData = pd.read_csv(f)
            solData = solData.drop(columns=["TrialID", "ObsNum", "Indicator"])
            solData = solData.rename(columns={"Prediction": "IsAlert"})
            f.close()


            finalData = pd.concat([solData, testData], ignore_index=False, sort=False, axis=1)
            testData = pd.concat([solData, testData], ignore_index=False, sort=False, axis=1)
            # finalData = pd.concat([solData, testData], ignore_index=False, sort=False, axis=1)


            finalData = pd.concat([trainData, finalData], axis=0)

            # finalData = trainData

            finalDataScale = finalData.iloc[:, 1:].astype(np.float)
            scaler = StandardScaler()
            scaler.fit(finalDataScale)
            finalData.iloc[:,1:] = scaler.transform(finalDataScale)

            finalData = finalData.iloc[np.random.randint(600000, size=10000),:]
            finalData.to_csv("fordReady.csv", index=False)


            
            f = open("./fordReady.csv")
            finalData = pd.read_csv(f)
            f.close()


            self.trainX, self.testX, self.trainY, self.testY = train_test_split(finalData.iloc[:,1:], finalData.iloc[:,0], test_size=0.3, random_state=100, shuffle=True)

            self.trainX = self.trainX.reset_index(drop=True)
            self.trainY = self.trainY.reset_index(drop=True)


            print(self.trainX.shape)
            print(self.trainY.shape)
            print(self.testX.shape)
            print(self.testY.shape) 
        elif(dataset == "otto"):
            f = open("../data/otto/train.csv")
            data = pd.read_csv(f)
            data = data.drop(columns=["id"])
            f.close()

            data.loc[data['target'] == "Class_1", 'target'] = 1
            data.loc[data['target'] == "Class_2", 'target'] = 2
            data.loc[data['target'] == "Class_3", 'target'] = 3
            data.loc[data['target'] == "Class_4", 'target'] = 4
            data.loc[data['target'] == "Class_5", 'target'] = 5
            data.loc[data['target'] == "Class_6", 'target'] = 6
            data.loc[data['target'] == "Class_7", 'target'] = 7
            data.loc[data['target'] == "Class_8", 'target'] = 8
            data.loc[data['target'] == "Class_9", 'target'] = 9

            data = data.iloc[np.random.randint(43314, size=10000),:]
            data.to_csv("ottoReady.csv", index=False)

            dataX = data.ix[:, data.columns != 'target']
            dataY = data.ix[:, data.columns == 'target']

            self.trainX, self.testX, self.trainY, self.testY = train_test_split(dataX,dataY,test_size = 0.3, random_state = 0)


            self.trainX = self.trainX.reset_index(drop=True)
            self.trainY = self.trainY.reset_index(drop=True)

            print(self.trainX.shape)
            print(self.trainY.shape)
            print(self.testX.shape)
            print(self.testY.shape)


    def plotExpDT(self, x):
        plt.plot(x, self.dtTrainTime, color="red")
        plt.xlabel("Max Depth")
        plt.ylabel("Training Time in seconds")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Training Time Vs Max Depth; Stay Alert Dataset")
        plt.savefig("./DT/" + self.dataset + "_DT-Traintime.png")
        
        plt.close()

        plt.plot(x, self.dtTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.dtTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.dtCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.xlabel("Max Depth")
        plt.ylabel("Accuracy (out of 1)")
        plt.title("Accuracy Vs Max Depth; Stay Alert Dataset")
        plt.savefig("./DT/" + self.dataset + "_DT-Accuracy.png")
        plt.close()

    def plotDecisionTreeLC(self, x, max_depth):
        plt.plot(x, self.dtTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.dtTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.dtCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy (out of 1)")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Accuracy vs Training Set Size; Stay Alert Dataset max_depth = " + str(max_depth))
        plt.savefig("./DT/" + self.dataset + "_DT-LearningCurves.png")
        plt.close()

    def resetDT(self):
        self.dtTrainAccuracy = []
        self.dtTestAccuracy = []
        self.dtCrossAccuracy = []
        self.dtTrainTime = []

    def logDT(self):
        print("self.dtTrainTime =  ", self.dtTrainTime)
        print("self.dtTrainAccuracy =  ", self.dtTrainAccuracy)
        print("self.dtCrossAccuracy =  ", self.dtCrossAccuracy)
        print("self.dtTestAccuracy = ", self.dtTestAccuracy)

    def expDecisionTree(self):      
        for depth in range(1,25):
            self.decisionTree(depth = depth, sampleSize = 1.0)
        self.plotExpDT(range(1,25))

        self.logDT()
        self.resetDT()

        for sampleSize in np.linspace(0.1, 1, num=10):
            self.decisionTree(depth=10, sampleSize = sampleSize)
        self.plotDecisionTreeLC(np.linspace(0.1, 1, num=10), 10)
        self.logDT()

    def decisionTree(self, depth = 2, sampleSize = 1.0):
        nSamples = int(round(len(self.trainX)) * sampleSize)
        rows = random.sample(range(len(self.trainX)), nSamples)

        trainX = self.trainX.ix[rows,]
        trainY = self.trainY.ix[rows,]

        startTrainingTime = time.time()
        self.dt = tree.DecisionTreeClassifier(max_depth=depth)
        self.dt.fit(trainX, trainY)
        endTrainingTime = time.time()
        print("DT - Max Depth %s" %(depth))
        trainingTime = endTrainingTime-startTrainingTime
        print("Time taken to train %s training samples is %s seconds" %(str(len(trainX)), str(trainingTime)))

        train_predict = self.dt.predict(trainX)
        test_predict = self.dt.predict(self.testX)
        trainAccuracy = accuracy_score(trainY, train_predict.round())
        testAccuracy = accuracy_score(self.testY, test_predict.round())
        validationAccuracy = np.mean(cross_val_score(self.dt, trainX, trainY, cv=3))
        print("Training Accuracy %s" %(str(trainAccuracy)))
        print("Validation Accuracy %s" %(str(validationAccuracy)))
        print("Testing Accuracy %s" %(str(testAccuracy)))
        print("=====================\n")

        self.dtTrainTime.append(trainingTime)
        self.dtTrainAccuracy.append(trainAccuracy)
        self.dtTestAccuracy.append(testAccuracy)
        self.dtCrossAccuracy.append(validationAccuracy)

    def plotExpBoost(self, x):
        plt.plot(x, self.boostTrainTime, color="red")
        plt.xlabel("Max Depth")
        plt.ylabel("Training Time in seconds")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Training Time Vs Max Depth; Stay Alert Dataset")
        plt.savefig("./Boost/" + self.dataset + "_AdaBoost-Traintime.png")
        plt.close()

        plt.plot(x, self.boostTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.boostTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.boostCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xlabel("Max Depth")
        plt.ylabel("Accuracy (out of 1)")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Accuracy Vs Max Depth; Stay Alert Dataset")
        plt.savefig("./Boost/" + self.dataset + "_AdaBoost-Accuracy.png")
        plt.close()

    def plotExpBoostEstimators(self, x):

        plt.plot(x, self.boostTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.boostTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.boostCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xlabel("n_estimators")
        plt.ylabel("Accuracy (out of 1)")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Accuracy Vs n_estimators [Stay Alert Dataset] [max_depth=13]")
        plt.savefig("./Boost/" + self.dataset + "_AdaBoost-estimator-Accuracy.png")
        plt.close()

    def plotBoostLC(self, x):
        plt.plot(x, self.boostTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.boostTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.boostCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy (out of 1)")
        plt.title("Stay Alert Dataset [max_depth=13, n_estimators=600]")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.savefig("./Boost/" + self.dataset + "_Boost-LearningCurves.png")
        plt.close()

    def resetBoost(self):
        self.boostTrainAccuracy = []
        self.boostCrossAccuracy = []
        self.boostTestAccuracy = []
        self.boostTrainTime = []

    def logBoosting(self):
        print("Training Time: ", self.boostTrainTime)
        print("Train Accuracy: ", self.boostTrainAccuracy)
        print("Cross Validation Accuracy: ", self.boostCrossAccuracy)
        print("Test Accuracy: ", self.boostTestAccuracy)

    def expBoosting(self):

        for depth in range(1,30,2):
            self.boosting(max_depth=depth)
        self.logBoosting()
        self.plotExpBoost(range(1,30,2))
        self.resetBoost()

        for n_estimators in range(50,601,50):
            self.boosting(n_estimators=n_estimators, max_depth=13)
        self.logBoosting()
        self.plotExpBoostEstimators(range(50,601,50))
        self.resetBoost()

        for sampleSize in np.linspace(0.1, 1, num=10):
            self.boosting(max_depth=13, n_estimators=600, sampleSize = sampleSize)
        self.logBoosting()
        self.plotBoostLC(np.linspace(0.1, 1, num=10))

    def boosting(self, max_depth = None, n_estimators = 50, algorithm = "SAMME.R", sampleSize = 1.0):
        nSamples = int(round(len(self.trainX)) * sampleSize)
        rows = random.sample(range(len(self.trainX)), nSamples)

        trainX = self.trainX.ix[rows,]
        trainY = self.trainY.ix[rows,]


        startTrainingTime = time.time()
        estimator = tree.DecisionTreeClassifier(max_depth = max_depth)
        self.boostingClassifier = AdaBoostClassifier( estimator, n_estimators= n_estimators,
                                 learning_rate=1.5, algorithm=algorithm)
        self.boostingClassifier.fit(trainX, trainY)
        endTrainingTime = time.time()
        trainingTime = endTrainingTime-startTrainingTime
        print("Boosting - Time taken to train %s training samples is %s seconds" %(str(len(trainX)), str(trainingTime)))



        train_predict = self.boostingClassifier.predict(trainX)
        test_predict = self.boostingClassifier.predict(self.testX)


        trainAccuracy = accuracy_score(trainY, train_predict.round())
        testAccuracy = accuracy_score(self.testY, test_predict.round())
        validationAccuracy = np.mean(cross_val_score(self.boostingClassifier, trainX, trainY, cv=3))
        print("Training Accuracy %s" %(str(trainAccuracy)))
        print("Validation Accuracy %s" %(str(validationAccuracy)))
        print("Testing Accuracy %s" %(str(testAccuracy)))
        print("=====================\n")

        self.boostTrainTime.append(trainingTime)
        self.boostTrainAccuracy.append(trainAccuracy)
        self.boostCrossAccuracy.append(validationAccuracy)
        self.boostTestAccuracy.append(testAccuracy)

    def plotExpKnn(self, x):
        #Training time
        plt.plot(x, self.knnTrainTime, color="red")
        plt.xlabel("k (n_neighbors)")
        plt.ylabel("Training Time in seconds")
        plt.xticks(range(1,26))
        plt.legend(loc='upper left')
        plt.title("Training Time Vs k [Stay Alert Dataset]")
        plt.savefig("./knn/" + self.dataset + "_knn-Traintime.png")
        plt.close()


        #Testing time for train and test samples
        plt.plot(x, self.knnTestTime_Train, color="red", label="Training Data")
        plt.plot(x, self.knnTestTime_Test, color="blue", label="Testing Data")
        plt.xlabel("k (n_neighbors)")
        plt.ylabel("Time in Seconds")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Testing Time Vs k [Stay Alert Dataset]")
        plt.savefig("./knn/" + self.dataset + "_knn-testTime.png")
        plt.close()

        
        #Accuracies over varying value of k        
        plt.plot(x, self.knnTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.knnTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.knnCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xlabel("k (n_neighbors)")
        plt.ylabel("Accuracy (out of 1)")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Accuracy Vs k [Stay Alert Dataset]")
        plt.savefig("./knn/" + self.dataset + "_knn-Accuracy.png")
        plt.close()

    def plotKnnLC(self, x):
        plt.plot(x, self.knnTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.knnTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.knnCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy (out of 1)")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Ford Dataset [k=1, distance_metric=Manhattan]")
        plt.savefig("./knn/" + self.dataset + "_knn-LearningCurves.png")
        plt.close()

    def logKnn(self):
        print("Training Time: ", self.knnTrainTime)
        print("Testing Time (Train):", self.knnTestTime_Train)
        print("Testing Time (Test):", self.knnTestTime_Test)
        print("Train Accuracy:", self.knnTrainAccuracy)
        print("Cross Validation Accuracy:", self.knnCrossAccuracy)
        print("Test Accuracy:", self.knnTestAccuracy)

    def resetKnn(self):
        self.knnTrainAccuracy = []
        self.knnTestAccuracy = []
        self.knnCrossAccuracy = []
        self.knnTrainTime = []
        self.knnTestTime_Train = []
        self.knnTestTime_Test = []

    def expKnn(self):
        for neighbors in range(1,40,2):
            self.KNN(n_neighbors = neighbors)

        print("K Accuracy & Time Testing and Training")
        self.logKnn()
        self.plotExpKnn(range(1,40,2))

        self.resetKnn()

        for distance_metric in ["hamming", "euclidean", "manhattan", "chebyshev"]:
            self.KNN(n_neighbors = 1, sampleSize=1.0, distance_metric = distance_metric)
            print("%s %s %s" %(distance_metric, str(self.knnCrossAccuracy[0]), str(self.knnTestAccuracy[0])))
            self.resetKnn()
            self.knn = None

        for sampleSize in np.linspace(0.1, 1, num=10):
            self.KNN(n_neighbors = 1, sampleSize = sampleSize, distance_metric = "manhattan")
        self.logKnn()
        self.plotKnnLC(np.linspace(0.1, 1, num=10))

    def KNN(self, n_neighbors = 1, sampleSize = 1.0, distance_metric="euclidean"):
        print("K =  %s" %(n_neighbors))
        nSamples = int(round(len(self.trainX)) * sampleSize)
        rows = random.sample(range(len(self.trainX)), nSamples)

        trainX = self.trainX.ix[rows,]
        trainY = self.trainY.ix[rows,]


        startTrainingTime = time.time()
        self.knn = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, n_jobs = -1, metric=distance_metric, p = 1)
        self.knn.fit(trainX, trainY)
        endTrainingTime = time.time()
        trainingTime = endTrainingTime-startTrainingTime
        print("Time taken to train %s training samples is %s seconds" %(str(len(trainX)), str(trainingTime)))
        self.knnTrainTime.append(trainingTime)

        startTrainingTime = time.time()
        train_predict = self.knn.predict(trainX)
        endTrainingTime = time.time()
        trainingTime = endTrainingTime-startTrainingTime
        print("Time taken to test %s training samples is %s seconds" %(str(len(trainX)), str(trainingTime)))
        self.knnTestTime_Train.append(trainingTime)
        

        startTrainingTime = time.time()
        test_predict = self.knn.predict(self.testX)
        endTrainingTime = time.time()
        trainingTime = endTrainingTime-startTrainingTime
        print("Time taken to train %s testing samples is %s seconds" %(str(len(self.testX)), str(trainingTime)))
        self.knnTestTime_Test.append(trainingTime)


        trainAccuracy = accuracy_score(trainY, train_predict.round())
        validationAccuracy = np.mean(cross_val_score(self.knn, trainX, trainY, cv=3))
        testAccuracy = accuracy_score(self.testY, test_predict.round())
        print("Training Accuracy %s" %(str(trainAccuracy)))
        print("Validation Accuracy %s" %(str(validationAccuracy)))
        print("Testing Accuracy %s" %(str(testAccuracy)))
        print("=====================\n")
        print("\n")    
        
        self.knnTrainAccuracy.append(trainAccuracy)
        self.knnCrossAccuracy.append(validationAccuracy)
        self.knnTestAccuracy.append(testAccuracy)

    def plotSvm(self): 
        plt.bar(['rbf', 'linear', 'poly'],[self.svmTrainTime['rbf'][0], self.svmTrainTime['linear'][0], self.svmTrainTime['poly'][0]], color='blue')
        plt.text(0 - 0.1, self.svmTrainTime["rbf"][0]  + 1, str(self.svmTrainTime['rbf'][0])[:5], color='black')
        plt.text(1 - 0.1, self.svmTrainTime["linear"][0]  + 1, str(self.svmTrainTime['linear'][0])[:5], color='black')
        plt.text(2 - 0.1, self.svmTrainTime["poly"][0]  + 1, str(self.svmTrainTime['poly'][0])[:5], color='black')

        plt.xlabel('Kernel')
        plt.ylabel('Training time in seconds')

        plt.title('Training Time Vs Kernel [Stay Alert Dataset]')

         
        # Create legend & Show graphic
        plt.legend(loc='upper left')
        plt.savefig("./svm/" + self.dataset + "_svm-kernelTrainTime.png")
        plt.close()

        barWidth = 0.25
         
        # set height of bar
        bars1 = [self.svmTrainAccuracy['rbf'][0], self.svmCrossAccuracy['rbf'][0], self.svmTestAccuracy['rbf'][0]]
        bars2 = [self.svmTrainAccuracy['linear'][0], self.svmCrossAccuracy['linear'][0], self.svmTestAccuracy['linear'][0]]
        bars3 = [self.svmTrainAccuracy['poly'][0], self.svmCrossAccuracy['poly'][0], self.svmTestAccuracy['poly'][0]]

         
        # Set position of bar on X axis
        r1 = np.arange(len(bars1))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]

        print(bars1, bars2, bars3)
        print(r1, r2, r3)
         
        # Make the plot
        plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Training Score')
        plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Cross Validation Score')
        plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Testing Score')
        plt.text(-0.1, bars1[0], str(bars1[0])[:5], color='black')
        plt.text(0.12, bars2[0], str(bars2[0])[:5], color='black')
        plt.text(0.38, bars3[0], str(bars3[0])[:5], color='black')

        plt.text(1 -0.1, bars1[1], str(bars1[1])[:5], color='black')
        plt.text(1 + 0.12, bars2[1] , str(bars2[1])[:5], color='black')
        plt.text(1 + 0.37, bars3[1]  , str(bars3[1])[:5], color='black')

        plt.text(2 -0.1, bars1[2], str(bars1[2])[:5], color='black')
        plt.text(2 + 0.12, bars2[2], str(bars2[2])[:5], color='black')
        plt.text(2 + 0.37, bars3[2], str(bars3[2])[:5], color='black')
         
        # Add xticks on the middle of the group bars
        plt.xlabel('Kernel', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(bars1))], ['rbf', 'linear', 'poly'])
         
        # Create legend & Show graphic
        plt.legend(loc='lower left')
        plt.title("Kernel Accuracies [Stay Alert Dataset]")
        plt.savefig("./svm/" + self.dataset + "_svm-kernelAccuracy.png")
        plt.close()

    def plotSvmLC(self, x):
        # plt.plot(x, self.svmTrainAccuracy['linear'], label="linear train score", color="red")
        # plt.plot(x, self.svmCrossAccuracy['linear'], label="linear cross validation score", color="green")
        # plt.plot(x, self.svmTestAccuracy['linear'], '--', label="linear test score", color="blue")
        
        # plt.plot(x, self.svmTrainAccuracy['poly'], label="poly train score", color="orange")
        # plt.plot(x, self.svmCrossAccuracy['poly'], label="poly cross validation score", color="brown")
        # plt.plot(x, self.svmTestAccuracy['poly'], '--', label="poly test score", color="yellow")

        plt.plot(x, self.svmTrainAccuracy['rbf'], label="rbf train score", color="red")
        plt.plot(x, self.svmCrossAccuracy['rbf'], label="rbf cross validation score", color="green")
        plt.plot(x, self.svmTestAccuracy['rbf'], '--', label="rbf test score", color="blue")
        plt.xlabel("Training Set Size (times of original training size)")
        plt.ylabel("Accuracy")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Stay Alert Dataset [kernel=rbf, C=1]")
        plt.savefig("./svm/" + self.dataset + "_svm-learningCurves.png")
        plt.close()

    def plotSVMPenalty(self, x):
        plt.plot(x, self.svmTrainAccuracy['linear'], label="linear train score", color="red")
        plt.plot(x, self.svmCrossAccuracy['linear'], '--', label="linear cross validation score", color="green")
        # plt.plot(x, self.svmTestAccuracy['linear'], '--', label="linear test score", color="blue")
        
        plt.plot(x, self.svmTrainAccuracy['poly'], label="poly train score", color="orange")
        plt.plot(x, self.svmCrossAccuracy['poly'], '--', label="poly cross validation score", color="brown")
        # plt.plot(x, self.svmTestAccuracy['poly'], '--', label="poly test score", color="yellow")

        plt.plot(x, self.svmTrainAccuracy['rbf'], label="rbf train score", color="black")
        plt.plot(x, self.svmCrossAccuracy['rbf'], '--', label="rbf cross validation score", color="purple")
        # plt.plot(x, self.svmTestAccuracy['rbf'], '--', label="rbf test score", color="brown")
        plt.xlabel("C")
        plt.ylabel("Accuracy")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.title("Stay Alert Dataset")
        plt.savefig("./svm/" + self.dataset + "_svm-penalty.png")
        plt.close()

    def logSvm(self):
        print("Train Time:", self.svmTrainTime)
        print("Training Score:", self.svmTrainAccuracy)
        print("Cross validation Score:", self.svmCrossAccuracy)
        print("Testing Score:", self.svmTestAccuracy)

    def resetSvm(self):
        self.svmTrainTime = {'linear': [], 'poly': [], 'rbf': []}
        self.svmTrainAccuracy = {'linear': [], 'poly': [], 'rbf': []}
        self.svmCrossAccuracy = {'linear': [], 'poly': [], 'rbf': []}
        self.svmTestAccuracy = {'linear': [], 'poly': [], 'rbf': []}

    def expSvm(self):
        for kernel in ['linear', 'poly', 'rbf']:
            print("Kernel = %s" %(kernel))
            self.SVM(kernel=kernel, sampleSize = 1.0)

        print("SVM Kernel times and Accuracies")
        self.logSvm()
        self.plotSvm()
        self.resetSvm()


        # x = []
        # for kernel in ['linear', 'poly', 'rbf']:
        #     print("Kernel = %s" %(kernel))
        #     c = 2 
        #     p = -2
        #     while( p <= 5):
        #         self.SVM(kernel=kernel, C = c**p)
        #         x.append(c**p)
        #         p = p + 1
        # self.logSvm()
        # self.plotSVMPenalty(x[:8])
        # self.resetSvm()

        # for kernel in ['linear', 'poly', 'rbf']:
        #     print("Kernel = %s" %(kernel))
        #     for sampleSize in np.linspace(0.1, 1, num=10):
        #         self.SVM(kernel=kernel, sampleSize = sampleSize)
        # self.logSvm()
        # self.svmTrainAccuracy = {'rbf': [0.8428571428571429, 0.8578571428571429, 0.8438095238095238, 0.8467857142857143, 0.8431428571428572, 0.8464285714285714, 0.8508163265306122, 0.84875, 0.8519047619047619, 0.8544285714285714], 'linear': [0.8042857142857143, 0.7692857142857142, 0.7938095238095239, 0.7796428571428572, 0.7951428571428572, 0.7866666666666666, 0.7914285714285715, 0.79375, 0.7922222222222223, 0.7935714285714286], 'poly': [0.76, 0.7978571428571428, 0.8014285714285714, 0.8092857142857143, 0.814, 0.814047619047619, 0.8126530612244898, 0.8258928571428571, 0.8255555555555556, 0.8242857142857143]}
        # self.svmCrossAccuracy = {'rbf': [0.7785603365002506, 0.7892777227541176, 0.7880952380952381, 0.8153539081332962, 0.8022838855646732, 0.8145174554146396, 0.8155133950600165, 0.818034201569389, 0.8212707976462594, 0.8214309511176174], 'linear': [0.7628969345707542, 0.7557170996804857, 0.7757248649389857, 0.7739285137778634, 0.7865691890040729, 0.7809483932195397, 0.7848984708122334, 0.7882124861717964, 0.7895221153219353, 0.7914295988943124], 'poly': [0.6885622359760291, 0.7249978601909932, 0.736655426604855, 0.7514254479077492, 0.7580045740423099, 0.7642857142857142, 0.7638766985393817, 0.7826751371130558, 0.7841258936757417, 0.7861442920253611]}
        # self.svmTestAccuracy = {'rbf': [0.768, 0.7906666666666666, 0.803, 0.801, 0.8023333333333333, 0.809, 0.8183333333333334, 0.8166666666666667, 0.8233333333333334, 0.8236666666666667], 'linear': [0.7663333333333333, 0.7583333333333333, 0.772, 0.7666666666666667, 0.782, 0.7706666666666667, 0.7746666666666666, 0.776, 0.7756666666666666, 0.7773333333333333], 'poly': [0.695, 0.7406666666666667, 0.745, 0.7613333333333333, 0.7573333333333333, 0.7646666666666667, 0.7733333333333333, 0.7773333333333333, 0.7843333333333333, 0.785]}
        # self.plotSvmLC(np.linspace(0.1, 1, num=10))
        self.resetSvm()
        for sampleSize in np.linspace(0.1, 1, num=10):
            self.SVM(kernel='rbf', sampleSize = sampleSize)
        self.plotSvmLC(np.linspace(0.1, 1, num=10))

    def SVM(self, kernel = 'rbf', sampleSize = 1.0, C = 1.0):
        
        nSamples = int(round(len(self.trainX)) * sampleSize)
        rows = random.sample(range(len(self.trainX)), nSamples)

        trainX = self.trainX.ix[rows,]
        trainY = self.trainY.ix[rows,]

        startTrainingTime = time.time()
        self.svm = SVC(gamma='auto', kernel = kernel)
        self.svm.fit(trainX, trainY)
        endTrainingTime = time.time()
        trainingTime = endTrainingTime - startTrainingTime
        self.svmTrainTime[kernel].append(trainingTime)
        print("Time taken to train %s training samples is %s seconds" %(str(len(trainX)), str(trainingTime)))


        train_predict = self.svm.predict(trainX)
        test_predict = self.svm.predict(self.testX)

        trainAccuracy = accuracy_score(trainY, train_predict.round())
        validationAccuracy = np.mean(cross_val_score(self.svm, trainX, trainY, cv=3))
        testAccuracy = accuracy_score(self.testY, test_predict.round())
        print("Training Accuracy %s" %(str(trainAccuracy)))
        print("Validation Accuracy %s" %(str(validationAccuracy)))
        print("Testing Accuracy %s" %(str(testAccuracy)))
        print("=====================\n")
        print("\n")

        self.svmTrainAccuracy[kernel].append(trainAccuracy)
        self.svmCrossAccuracy[kernel].append(validationAccuracy)
        self.svmTestAccuracy[kernel].append(testAccuracy)

    def logMlpc(self):
        print("Train time", self.mlpcTrainTime)
        print("Train Accuracy", self.mlpcTrainAccuracy)
        print("Cross Validation Accuracy", self.mlpcCrossAccuracy)
        print("Test Accuracy", self.mlpcTestAccuracy)

    def resetMlpc(self):
        self.mlpcTrainTime = []
        self.mlpcTrainAccuracy = []
        self.mlpcCrossAccuracy = []
        self.mlpcTestAccuracy = []

    def plotMlpcNeurons(self, x):
        plt.plot(x, self.mlpcTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.mlpcTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.mlpcCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xlabel("Num of Neurons")
        plt.ylabel("Accuracy (out of 1)")
        plt.title("Accuracy Vs # Neurons for 1 layer [Stay Alert Dataset]")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.savefig("./mlpc/" + self.dataset + "_mlpc-NumNeurons.png")
        plt.close()

    def plotMlpcLayers(self, x):
        plt.plot(x, self.mlpcTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.mlpcTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.mlpcCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xlabel("Num of Layers")
        plt.ylabel("Accuracy (out of 1)")
        plt.title("Accuracy Vs #Layers [Stay Alert Dataset]")
        plt.xticks(x)
        plt.legend(loc='upper left')
        plt.savefig("./mlpc/" + self.dataset + "_mlpc-NumLayers.png")
        plt.close()

    def plotMlpcLC(self, x):
        plt.plot(x, self.mlpcTrainAccuracy, label="Training Score" ,color="red")
        plt.plot(x, self.mlpcTestAccuracy, '--', label="Testing Score" ,color="blue")
        plt.plot(x, self.mlpcCrossAccuracy, label="Cross Validation Score" ,color="green")
        plt.xlabel("Training Set Size (times of original training set)")
        plt.ylabel("Accuracy (out of 1)")
        plt.xticks(x)
        plt.title("Stay Alert Dataset [#Neurons=15, #Layers=1, func='relu']")
        plt.legend(loc='upper left')
        plt.savefig("./mlpc/" + self.dataset + "_mlpc-LearningCurves.png")
        plt.close()

    def expMlpc(self):
        # for numNeurons in range(1,40,2):
        #     self.MLPC(numNeurons = numNeurons)
        
        # print("Vs #Neurons")
        # self.logMlpc()
        # self.plotMlpcNeurons(range(1,40,2))
        # self.resetMlpc()

        # for numLayers in range(1,10):
        #     self.MLPC(numNeurons = 1, numLayers = numLayers)

        # print("Vs #Layers")
        # self.logMlpc()
        # self.plotMlpcLayers(range(1,10))
        # self.resetMlpc()

        # activationFunctions = ["identity", "logistic", "tanh", "relu"]
        # for actFunction in activationFunctions:
        #     self.MLPC(numNeurons = 15, activationFunction = actFunction)
        # self.logMlpc()
        # self.resetMlpc()

        # for sampleSize in np.linspace(0.1, 1, num=10):
        self.MLPC(numNeurons=15, numLayers = 1)
        # self.plotMlpcLC(np.linspace(0.1, 1, num=10))
        self.logMlpc()

    def MLPC(self, numNeurons = 1, numLayers = 1, sampleSize = 1.0, activationFunction = "relu"):
        print(" # Layers = %s, # Neurons = %s" %(str(numLayers), str(numNeurons)))
        nSamples = int(round(len(self.trainX)) * sampleSize)
        rows = random.sample(range(len(self.trainX)), nSamples)

        trainX = self.trainX.ix[rows,]
        trainY = self.trainY.ix[rows,]

        startTrainingTime = time.time()
        hidden_layer_sizes = tuple([numNeurons]*numLayers)
        self.mlpc = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
        self.mlpc.fit(trainX, trainY)
        endTrainingTime = time.time()
        trainingTime = endTrainingTime - startTrainingTime
        self.mlpcTrainTime.append(trainingTime)
        print("Time taken to train %s training samples is %s seconds" %(str(len(trainX)), str(trainingTime)))

        train_predict = self.mlpc.predict(trainX)
        test_predict = self.mlpc.predict(self.testX)

        trainAccuracy = accuracy_score(trainY, train_predict.round())
        validationAccuracy = np.mean(cross_val_score(self.mlpc, trainX, trainY, cv=3))
        testAccuracy = accuracy_score(self.testY, test_predict.round())
        print("Training Accuracy %s" %(str(trainAccuracy)))
        print("Validation Accuracy %s" %(str(validationAccuracy)))
        print("Testing Accuracy %s" %(str(testAccuracy)))
        print("=====================\n")
        print("\n")

        self.mlpcTrainAccuracy.append(trainAccuracy)
        self.mlpcCrossAccuracy.append(validationAccuracy)
        self.mlpcTestAccuracy.append(testAccuracy)



if __name__ == "__main__":
    # change dataset to "otto" to run experiments on otto dataset
    expInstance = SupervisedLearning(dataset = "otto")
    expInstance.runExperiment()
