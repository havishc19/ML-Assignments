import matplotlib.pyplot as plt
import numpy as np
from math import log

class PlotExp:
    def __init__(self):

       # self.readFile(name="small")
       # self.plotStuff(name="small")
       # self.readFile(name="large")
       # self.plotStuff(name="large")

       # self.plotQLearning(name="small")
       self.plotQLearning(name="large")


    def plotQLearning(self, name):
        # self.getEpsilonVals("epsilonRewards", name, "Total Rewards", "Tota Reward Vs #Iterations [alpha=0.5]")
        # self.getEpsilonVals("epsilontimes", name, "Total Time in ms", "Total time Vs #Iterations [alpha=0.5]")
        # self.getEpsilonVals("epsilonDelta", name, "Delta", "Delta Vs #Iterations")

        # self.getLearnerVals("learningRewards", name, "Total Rewards", "Total Reward Vs #Iterations")
        # self.getLearnerVals("learningTimes", name, "Total Time in ms", "Total time Vs #Iterations")

        self.getLearnerVals("gammaRewards", name, "Total Rewards", "Total Reward Vs #Iterations [alpha=0.5, e=0.1]")
        self.getLearnerVals("gammaTimes", name, "Total Time in ms", "Total time Vs #Iterations [alpha=0.5, e=0.1]")

   

    def readFile(self, name = "small"):
        f = open(name)
        lines = f.readlines()
        lines = map( lambda x:  x[:-1], lines)
        self.iterations = map(lambda x: int(x), lines[2].split(",")[1:])
        self.valueIterSteps = map(lambda x: int(x), lines[3].split(",")[1:])
        self.policyIterSteps = map(lambda x: int(x), lines[4].split(",")[1:])
        self.qLearningSteps = map(lambda x: int(x), lines[5].split(",")[1:])

        # print(self.valueIterSteps)
        # print("\n")
        # print(self.policyIterSteps)
        # print("\n")
        # print(self.qLearningSteps)

        self.valueIterTime = map(lambda x: int(x), lines[10].split(",")[1:])
        self.policyIterTime = map(lambda x: int(x), lines[11].split(",")[1:])
        self.qLearningTime = map(lambda x: int(x), lines[12].split(",")[1:])

        self.valueIterRewards = map(lambda x: float(x), lines[17].split(",")[1:])
        self.policyIterRewards = map(lambda x: float(x), lines[18].split(",")[1:])
        self.qLearningRewards = map(lambda x: float(x), lines[19].split(",")[1:])

        print("==========")
        print(self.valueIterRewards)
        print("==========")
        print(self.policyIterRewards)
        print("==========")
        print(self.qLearningRewards)
        print("==========")

    def plotTemplate(self, x, a, b, c, labelA, labelB, labelC, xLabel, ylabel, title, fileName):
        # plt.plot(x, a, '.-', label=labelA ,color="red")
        # plt.plot(x, b, '.-', label=labelB ,color="green")
        # plt.plot(x, c, '.-', label=labelC ,color="blue")
        plt.plot(x, a,  label=labelA ,color="red")
        plt.plot(x, b,  '--', label=labelB ,color="green")
        plt.plot(x, c,  label=labelC ,color="blue")
        # plt.vlines(self.find(a), min(a), max(a), linestyle=':', color="red", linewidth = 2)
        # plt.vlines(self.find(b), min(b), max(b), linestyle=':', color="green", linewidth = 2)
        # plt.vlines(self.find(c), min(c), max(c), linestyle=':', color="blue", linewidth = 2)
        plt.xticks(np.arange(1, x[-1] + 1, 50))
        plt.xlabel(xLabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='best')
        plt.savefig("./plots/" + fileName)
        plt.close()

    def plotEpsilonRewards(self, x, fileName, name, ylabel, title):
        # plt.plot(x, self.epsilonRewards["0.1"], '.-', color="red")
        plt.plot(x, self.epsilonRewards["0.1"], '-', label="Epsilon = 0.1", color="red")
        plt.plot(x, self.epsilonRewards["0.2"], '-', label="Epsilon = 0.2", color="green")
        plt.plot(x, self.epsilonRewards["0.3"], '-', label="Epsilon = 0.3", color="blue")
        plt.plot(x, self.epsilonRewards["0.5"], '-', label="Epsilon = 0.5", color="black")
        plt.xticks(np.arange(1, len(x)+1,50))
        plt.xlabel("# Iterations")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='best')
        plt.savefig("./plots/" + name + "/" + fileName + ".png")
        plt.close()

    def getEpsilonVals(self, fileName, name, ylabel, title):
        iterations = 500
        f = open(fileName)
        lines = f.readlines()[0]
        temp = map(lambda x: float(x), lines[:-1].split(","))
        self.epsilonRewards = {"0.1": [], "0.2": [], "0.3": [], "0.5": []}
        self.epsilonRewards["0.1"] = temp[0:iterations]
        self.epsilonRewards["0.2"] = temp[iterations:2*iterations]
        self.epsilonRewards["0.3"] = temp[2*iterations:3*iterations]
        self.epsilonRewards["0.5"] = temp[3*iterations:4*iterations]

        self.plotEpsilonRewards(range(1,iterations+1), fileName, name, ylabel, title)

    def plotLearnerVals(self, x, fileName, name, ylabel, title):
        plt.plot(x, self.epsilonRewards["0.1"], '-', label="Gamma = 0.1", color="red")
        plt.plot(x, self.epsilonRewards["0.5"], '-', label="Gamma = 0.5", color="green")
        plt.plot(x, self.epsilonRewards["0.9"], '-', label="Gamma = 0.99", color="blue")
        plt.xticks(np.arange(1, len(x)+1,50))
        plt.xlabel("# Iterations")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='best')
        plt.savefig("./plots/" + name + "/" + fileName + ".png")
        plt.close()


    def getLearnerVals(self, fileName, name, ylabel, title):
        iterations = 500
        f = open(fileName)
        lines = f.readlines()[0]
        temp = map(lambda x: float(x), lines[:-1].split(","))
        self.epsilonRewards = {"0.1": [], "0.5": [], "0.9": []}
        self.epsilonRewards["0.1"] = temp[0:iterations]
        self.epsilonRewards["0.5"] = temp[iterations:2*iterations]
        self.epsilonRewards["0.9"] = temp[2*iterations:3*iterations]

        self.plotLearnerVals(range(1,iterations+1), fileName, name, ylabel, title)




    def plotStuff(self, name="small"):


        self.plotTemplate(self.iterations, self.valueIterRewards, self.policyIterRewards, self.qLearningRewards, "Value", "Policy", "Q Learning", "# Iterations", "Rewards", "Rewards Vs # Iterations", name + "/rewards.png")

        self.plotTemplate(self.iterations, self.valueIterSteps, self.policyIterSteps, self.qLearningSteps, "Value", "Policy", "Q Learning", "# Iterations", "Actions/Steps", "Actions Vs # Iterations", name + "/actions.png")

        self.plotTemplate(self.iterations, self.valueIterTime, self.policyIterTime, self.qLearningTime, "Value", "Policy", "Q Learning", "# Iterations", "Time in ms", "Time Vs # Iterations", name + "/time.png")


        


if __name__ == "__main__":
    plotExp = PlotExp()

