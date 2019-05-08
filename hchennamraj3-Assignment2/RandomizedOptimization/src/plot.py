import matplotlib.pyplot as plt
import numpy as np
from math import log

class PlotResults:
    def __init__(self):
        self.rh_loc = "../Optimization_Results/2019-03-02/ford_results_rhc.csv"
        self.sa_cooling = "../Optimization_Results/2019-03-03/ford_results_sa_varyingCooling.csv"
        self.sa_temp = "../Optimization_Results/2019-03-03/ford_results_sa_varyingTemp.csv"
        self.ga = "../Optimization_Results/2019-03-01/ford_results_ga.csv"

        self.ts = "../Optimization_Results/2019-03-01/travelingsalesman_results.csv"
        self.knapsack = "../Optimization_Results/2019-03-01/knapsack_results.csv"
        self.continuousPeaks = "../Optimization_Results/2019-03-01/continuousPeaks_results.csv"

        self.iterations = [10, 100, 500, 1000, 2500, 5000]
        self.iterationsRHC = [10, 100, 500, 1000, 2500, 3000, 3500, 4000, 4500, 5000]
        self.iterationsTemp = ["10", "100", "500", "1000", "2500", "5000"]
        self.iterations1 = [10, 100, 1000, 5000, 10000, 25000, 50000, 100000, 250000]
        self.coolingExp = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
        self.temp = [10,100,1000,10000,100000,1000000,10000000]

        self.rhTrainAccuracy = []
        self.rhTestAccuracy  = []
        self.rhTrainTime = []
        self.rhTestTime = []

        self.saTrainAccuracy = {"0.05": [], "0.2": [], "0.35": [], "0.5": [], "0.65": [], "0.8": [], "0.95": [], "1.0E7": [], "1.0E9": [], "1.0E11": []}
        self.saTestAccuracy = {"0.05": [], "0.2": [], "0.35": [], "0.5": [], "0.65": [], "0.8": [], "0.95": [], "1.0E7": [], "1.0E9": [], "1.0E11": []}
        self.saTrainTime = {"0.05": [], "0.2": [], "0.35": [], "0.5": [], "0.65": [], "0.8": [], "0.95": [], "1.0E7": [], "1.0E9": [], "1.0E11": []}
        self.saTestTime = {"0.05": [], "0.2": [], "0.35": [], "0.5": [], "0.65": [], "0.8": [], "0.95": [], "1.0E7": [], "1.0E9": [], "1.0E11": []}

        
        self.gaTrainAccuracy = {"10,5,2": [], "25,10,5": [], "50,30,10": [], "75,40,15": [], "100,50,20": [], "125,60,25": []}
        self.gaTrainTime = {"10,5,2": [], "25,10,5": [], "50,30,10": [], "75,40,15": [], "100,50,20": [], "125,60,25": []}
        self.gaTestAccuracy = {"10,5,2": [], "25,10,5": [], "50,30,10": [], "75,40,15": [], "100,50,20": [], "125,60,25": []}
        self.gaTestTime = {"10,5,2": [], "25,10,5": [], "50,30,10": [], "75,40,15": [], "100,50,20": [], "125,60,25": []}


        self.tsFitnessScore = {"rhc": [], "sa": [], "ga": [], "mimic": []}
        self.tsTrainingTime = {"rhc": [], "sa": [], "ga": [], "mimic": []}

        self.knFitnessScore = {"rhc": [], "sa": [], "ga": [], "mimic": []}
        self.knTrainingTime = {"rhc": [], "sa": [], "ga": [], "mimic": []}

        self.cpFitnessScore = {"rhc": [], "sa": [], "ga": [], "mimic": []}
        self.cpTrainingTime = {"rhc": [], "sa": [], "ga": [], "mimic": []}



        # self.plotRH()
        # self.plotSA()
        self.plotGA()

        # self.plotTS()
        # self.plotKnapsack()
        # self.plotContinuousPeaks()

    def resetSA(self):
        self.saTrainAccuracy = {"0.05": [], "0.2": [], "0.35": [], "0.5": [], "0.65": [], "0.8": [], "0.95": [], "1.0E7": [], "1.0E9": [], "1.0E11": []}
        self.saTestAccuracy = {"0.05": [], "0.2": [], "0.35": [], "0.5": [], "0.65": [], "0.8": [], "0.95": [], "1.0E7": [], "1.0E9": [], "1.0E11": []}
        self.saTrainTime = {"0.05": [], "0.2": [], "0.35": [], "0.5": [], "0.65": [], "0.8": [], "0.95": [], "1.0E7": [], "1.0E9": [], "1.0E11": []}
        self.saTestTime = {"0.05": [], "0.2": [], "0.35": [], "0.5": [], "0.65": [], "0.8": [], "0.95": [], "1.0E7": [], "1.0E9": [], "1.0E11": []}

    def plotRHAccuracies(self, x, xTicks):
        plt.plot(x, self.rhTrainAccuracy, '.-', label="Training Score" ,color="red")
        plt.plot(x, self.rhTestAccuracy, '.-',label="Testing Score" ,color="blue")
        horiz_line_data = np.array([83.128 for i in xrange(len(x))])
        plt.plot(x, horiz_line_data, '--', color="black", label="Backprop NN") 
        plt.xlabel("# Iterations")
        plt.ylabel("Accuracy(%)")
        plt.xticks(x, xTicks)
        plt.title("Accuracy Vs Iterations for Random Hill Climbing")
        plt.legend(loc='best')
        plt.savefig("./plots/rh/accuracies.png")
        plt.close()

    def plotRHTimes(self, x, xTicks):
        plt.plot(x, self.rhTrainTime, label="Training Time" ,color="red")
        plt.plot(x, self.rhTestTime, '--', label="Testing Time" ,color="blue")
        plt.xlabel("# Iterations")
        plt.ylabel("Time in seconds")
        plt.xticks(x, xTicks)
        plt.title("Time Vs Iterations for Random Hill Climbing")
        plt.legend(loc='best')
        plt.savefig("./plots/rh/time.png")
        plt.close()

    def plotRH(self):
        # self.rhTrainAccuracy, self.rhTestAccuracy, self.rhTrainTime, self.rhTestTime = self.extractData(self.rh_loc, 3, 3, -3, -1)
        self.rhTrainAccuracy = ['50.357', '58.686', '71.800', '77.471', '81.486', '83.929', '84.109', '84.529', '84.826', '84.629']
        self.rhTestAccuracy = ['49.633', '57.800', '70.533', '75.833', '78.767', '80.600', '81.900', '81.300', '81.760', '81.600']
        # self.rhTrainTime = ['0.429', '2.742', '12.483', '26.217', '64.647', '129.282']
        # self.rhTestTime = ['0.031', '0.013', '0.010', '0.011', '0.010', '0.010']
        self.plotRHAccuracies(range(len(self.iterationsRHC)),self.iterationsRHC)
        # self.plotRHTimes(range(len(self.iterationsRHC)),self.iterationsRHC)

    def extractData(self, fileLoc, trainAccuracyIndex, testAccuracyIndex, trainTimeIndex, testTimeIndex):
        f = open(fileLoc)
        lines = f.readlines()
        i = 0;
        a = []
        b = []
        c = []
        d = []
        while(i < len(lines)):
            trainingData = lines[i][:-1]
            testingData = lines[i+1][:-1]
            tempTrain = trainingData.split(",")
            tempTest = testingData.split(",")
            # iterations = str(tempTrain[1])
            trainAccuracy = str(tempTrain[trainAccuracyIndex])
            testAccuracy = str(tempTest[testAccuracyIndex])
            trainTime = str(tempTrain[trainTimeIndex])
            testTime = str(tempTest[testTimeIndex])

            a.append(trainAccuracy)
            b.append(testAccuracy)
            c.append(trainTime)
            d.append(testTime)
            # print(trainAccuracy)
            # print(testAccuracy)
            # print("=====")
            i = i + 2
        return a,b,c,d

    def plotSAAccuracies(self, x, xTicks, xLabel):
        plt.plot(x, self.saTrainAccuracy["0.35"], label="Train, C=0.8" ,color="orange")
        plt.plot(x, self.saTrainAccuracy["0.65"], label="Train, C=0.65" ,color="black")
        plt.plot(x, self.saTrainAccuracy["0.8"], label="Train, C=0.35" ,color="grey")
        plt.plot(x, self.saTestAccuracy["0.35"], '--', label="Test, C=0.8" ,color="red")
        plt.plot(x, self.saTestAccuracy["0.65"], '--', label="Test, C=0.65" ,color="blue")
        plt.plot(x, self.saTestAccuracy["0.8"], '--', label="Test, C=0.35" ,color="green")
        horiz_line_data = np.array([83.128 for i in xrange(len(x))])
        plt.plot(x, horiz_line_data, ':', color="brown", label="Backprop NN") 
        plt.xlabel(xLabel)
        plt.ylabel("Accuracy(%)")
        plt.xticks(x, xTicks)
        plt.title("Accuracy Vs Iterations")
        plt.legend(loc='best')
        plt.savefig("./plots/sa/" + xLabel + "_accuracies.png")
        plt.close()


        plt.plot(x, self.saTrainTime["0.35"], '.-', label="C=0.35" ,color="orange")
        plt.plot(x, self.saTrainTime["0.65"], '.-', label="C=0.65" ,color="black")
        plt.plot(x, self.saTrainTime["0.8"], '.-', label="C=0.8" ,color="grey")
        plt.xlabel(xLabel)
        plt.ylabel("Training Time (sec)")
        plt.xticks(x, xTicks)
        plt.title("Training time Vs Iterations")
        plt.legend(loc='best')
        plt.savefig("./plots/sa/" + xLabel + "_time.png")
        plt.close()




    def plotSAAccuracies1(self, x, xTicks, xLabel):
        plt.plot(x, self.saTrainAccuracy["1.0E7"], label="Train, T=1.0E7" ,color="orange")
        plt.plot(x, self.saTrainAccuracy["1.0E9"], label="Train, T=1.0E9" ,color="black")
        plt.plot(x, self.saTrainAccuracy["1.0E11"], label="Train, T=1.0E11" ,color="grey")
        plt.plot(x, self.saTestAccuracy["1.0E7"], '--', label="Test, T=1.0E7" ,color="red")
        plt.plot(x, self.saTestAccuracy["1.0E9"], '--', label="Test, T=1.0E9" ,color="blue")
        plt.plot(x, self.saTestAccuracy["1.0E11"], '--', label="Test, T=1.0E11" ,color="green")
        horiz_line_data = np.array([83.128 for i in xrange(len(x))])
        plt.plot(x, horiz_line_data, ':', color="brown", label="Backprop NN") 
        plt.xlabel(xLabel)
        plt.ylabel("Accuracy(%)")
        plt.xticks(x, xTicks)
        plt.title("Accuracy Vs Iterations")
        plt.legend(loc='best')
        plt.savefig("./plots/sa/" + xLabel + "_accuracies.png")
        plt.close()


        plt.plot(x, self.saTrainTime["1.0E7"], '.-', label="T=1.0E7" ,color="orange")
        plt.plot(x, self.saTrainTime["1.0E9"], '.-', label="T=1.0E9" ,color="black")
        plt.plot(x, self.saTrainAccuracy["1.0E11"], '.-', label="T=1.0E11" ,color="grey")
        plt.xlabel(xLabel)
        plt.ylabel("Training Time (sec)")
        plt.xticks(x, xTicks)
        plt.title("Training time Vs Iterations")
        plt.legend(loc='best')
        plt.savefig("./plots/sa/" + xLabel + "_time.png")
        plt.close()



    def extractSaData(self, fileLoc, keyIndex):
        f = open(fileLoc)
        lines = f.readlines()
        i = 0;
        while(i < len(lines)):
            trainingData = lines[i][:-1]
            testingData = lines[i+1][:-1]
            tempTrain = trainingData.split(",")
            tempTest = testingData.split(",")
            key = str(tempTrain[keyIndex])
            print(key, trainingData)
            self.saTrainAccuracy[key].append(tempTrain[-5])
            self.saTestAccuracy[key].append(tempTest[-5])
            self.saTrainTime[key].append(tempTrain[-3])
            self.saTestTime[key].append(tempTrain[-1])
            i = i + 2

    def plotSA(self):
        print("SA")
        self.extractSaData(self.sa_cooling, 2)
        print(self.saTestAccuracy)
        print(self.saTrainAccuracy)
        self.plotSAAccuracies(range(len(self.iterations)),self.iterations, "CoolingExp")

        self.resetSA()

        self.extractSaData(self.sa_temp, 3)

        self.plotSAAccuracies1(range(len(self.iterations)),self.iterations, "Temperature")


    def extractGaData(self,):
        f = open(self.ga)
        lines = f.readlines()
        i = 0;
        while(i < len(lines)):
            trainingData = lines[i][:-1]
            testingData = lines[i+1][:-1]
            tempTrain = trainingData.split(",")
            tempTest = testingData.split(",")
            key = tempTrain[2] + "," + tempTrain[3] + "," + tempTrain[4]
            self.gaTrainAccuracy[key].append(tempTrain[6])
            self.gaTestAccuracy[key].append(tempTest[6])
            self.gaTrainTime[key].append(tempTrain[-3])
            self.gaTestTime[key].append(tempTrain[-1])
            i = i + 2

    def plotGA1(self,x, xTicks):
        plt.plot(x, self.gaTrainAccuracy["10,5,2"], '.-', label="10,5,2" ,color="red")
        plt.plot(x, self.gaTrainAccuracy["25,10,5"], '.-', label="25,10,5" ,color="orange")
        plt.plot(x, self.gaTrainAccuracy["50,30,10"], '.-', label="50,30,10" ,color="blue")
        plt.plot(x, self.gaTrainAccuracy["75,40,15"], '.-', label="75,40,15" ,color="green")
        plt.plot(x, self.gaTrainAccuracy["100,50,20"], '.-', label="100,50,20" ,color="black")
        plt.plot(x, self.gaTrainAccuracy["125,60,25"], '.-', label="125,60,25" ,color="purple")
        horiz_line_data = np.array([83.128 for i in xrange(len(x))])
        plt.plot(x, horiz_line_data, '--', color="brown", label="Backprop NN") 
        plt.xticks(x, xTicks)
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy(%)")
        plt.title("Training Accuracy Vs Iterations")
        plt.legend(loc='best')
        plt.savefig("./plots/ga/train_accuracies.png")
        plt.close()


        plt.plot(x, self.gaTestAccuracy["10,5,2"], '.-', label="10,5,2" ,color="red")
        plt.plot(x, self.gaTestAccuracy["25,10,5"], '.-', label="25,10,5" ,color="orange")
        plt.plot(x, self.gaTestAccuracy["50,30,10"], '.-', label="50,30,10" ,color="blue")
        plt.plot(x, self.gaTestAccuracy["75,40,15"], '.-', label="75,40,15" ,color="green")
        plt.plot(x, self.gaTestAccuracy["100,50,20"], '.-', label="100,50,20" ,color="black")
        plt.plot(x, self.gaTestAccuracy["125,60,25"], '.-', label="125,60,25" ,color="purple")
        horiz_line_data = np.array([83.128 for i in xrange(len(x))])
        plt.plot(x, horiz_line_data, '--', color="brown", label="Backprop NN") 
        plt.xticks(x, xTicks)
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy(%)")
        plt.title("Testing Accuracy Vs Iterations")
        plt.legend(loc='best')
        plt.savefig("./plots/ga/test_accuracies.png")
        plt.close()

        plt.plot(x, self.gaTrainTime["10,5,2"], '.-', label="10,5,2" ,color="red")
        plt.plot(x, self.gaTrainTime["25,10,5"], '.-', label="25,10,5" ,color="orange")
        plt.plot(x, self.gaTrainTime["50,30,10"], '.-', label="50,30,10" ,color="blue")
        plt.plot(x, self.gaTrainTime["75,40,15"], '.-', label="75,40,15" ,color="green")
        plt.plot(x, self.gaTrainTime["100,50,20"], '.-', label="100,50,20" ,color="black")
        plt.plot(x, self.gaTrainTime["125,60,25"], '.-', label="125,60,25" ,color="purple")
        plt.xticks(x, xTicks)
        plt.xlabel("Iterations")
        plt.ylabel("Training Time in seconds")
        plt.title("Training Time Vs Iterations")
        plt.legend(loc='best')
        plt.savefig("./plots/ga/train_times.png")
        plt.close()

    def plotGA(self):
        self.extractGaData()
        print(self.gaTrainAccuracy["100,50,20"])
        print(self.gaTestAccuracy["100,50,20"])
        self.plotGA1(range(len(self.iterations)), self.iterations)

    def extractData1(self, fileLoc):
        f = open(fileLoc)
        lines = f.readlines()
        a = {"rhc": [], "sa": [], "ga": [], "mimic": []}
        b = {"rhc": [], "sa": [], "ga": [], "mimic": []}
        for line in lines:
            tempLine = line[:-1].split(",")
            temp = tempLine
            a[temp[0]].append(temp[2])
            a[temp[4]].append(temp[6])
            a[temp[8]].append(temp[10])
            a[temp[12]].append(temp[14])

            b[temp[0]].append(log(float(temp[3])))
            b[temp[4]].append(log(float(temp[7])))
            b[temp[8]].append(log(float(temp[11])))
            b[temp[12]].append(log(float(temp[15])))
        return a,b

    def plotProblem(self, fitnessScore, trainingTime, x, title, folder, xTicks):
        plt.plot(x, fitnessScore["rhc"], '.-', label="rhc" ,color="red")
        plt.plot(x, fitnessScore["sa"], '.-', label="sa" ,color="orange")
        plt.plot(x, fitnessScore["ga"], '.-', label="ga" ,color="blue")
        plt.plot(x, fitnessScore["mimic"], '.-', label="mimic" ,color="green")
        plt.xticks(x, xTicks)
        plt.xlabel("# Iterations")
        plt.ylabel("Fitness Score")
        plt.title("Fitness Score Vs Iterations [" + title + "]")
        plt.legend(loc='best')
        plt.savefig("./plots/"  + folder + "/accuracies.png")
        plt.close()


        plt.plot(x, trainingTime["rhc"], '.-', label="rhc" ,color="red")
        plt.plot(x, trainingTime["sa"], '.-', label="sa" ,color="orange")
        plt.plot(x, trainingTime["ga"], '.-', label="ga" ,color="blue")
        plt.plot(x, trainingTime["mimic"], '.-', label="mimic" ,color="green")
        plt.xticks(x, xTicks)
        plt.xlabel("# Iterations")
        plt.ylabel("Training Time in log")
        plt.title("Training Time Vs Iterations [" + title + "]")
        plt.legend(loc='best')
        plt.savefig("./plots/" + folder + "/time.png")
        plt.close()

    def plotTS(self):
        self.tsFitnessScore, self.tsTrainingTime = self.extractData1(self.ts)
        # self.plotProblem(self.tsFitnessScore, self.tsTrainingTime, self.iterations1, "Travelling Salesman", "ts")
        self.plotProblem(self.tsFitnessScore, self.tsTrainingTime, range(len(self.iterations1)), "Travelling Salesman", "ts", self.iterations1)

    def plotKnapsack(self):
        self.knFitnessScore, self.knTrainingTime = self.extractData1(self.knapsack)
        print(self.knFitnessScore)
        print(self.iterations1)
        self.plotProblem(self.knFitnessScore, self.knTrainingTime, range(len(self.iterations)), "Knapsack", "knapsack", self.iterations)

    def plotContinuousPeaks(self):
        self.cpFitnessScore, self.cpTrainingTime = self.extractData1(self.continuousPeaks)
        self.plotProblem(self.cpFitnessScore, self.cpTrainingTime, range(len(self.iterations1)), "Continuous Peaks", "cPeaks", self.iterations1)



if __name__ == "__main__":
    plotInstance = PlotResults()

