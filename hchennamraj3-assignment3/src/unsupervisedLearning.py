import numpy as np
import pandas as pd
import time
import random
import os

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, silhouette_samples
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import random_projection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.cluster import adjusted_mutual_info_score
import scipy


class UnsupervisedLearning:
    def __init__(self, dataset):
        self.dataset = dataset
        self.datasetTitle = {"ford": "Stay Alert", "otto": "Otto"}
        self.pcaBest = 15
        self.icaBest = 5
        self.rpBest = 18
        self.faBest = 11
        if(self.dataset == "otto"):
            self.pcaBest = 25
            self.icaBest = 38
            self.rpBest = 42
            self.faBest = 30

    def runExperiment(self):
        self.loadData()
        # self.plotClusterDistribution(k=14, clusterAlgo = "kmeans")
        # self.plotClusterDistribution(14)
        # self.runKmeans()
        # self.runEM()
        # self.runPCA()
        # self.loadData()
        # self.runICA()
        # self.loadData()
        # self.runRP()
        # self.loadData
        # self.runFA()
        # self.kMeansExp()
        # self.loadData()
        self.EMExp()
        # self.expMlpc()
        # self.expLast()
        # self.loadData()
        # self.expLast(bow="1")
        # self.reconstructionError()

    def calcError(self, dim, data):
        dim.fit(data)
        data_transform = dim.transform(data)
        data_projected = dim.inverse_transform(data_transform)
        print((((data - data_projected) ** 2).mean()).mean())

    def calcError1(self, dim, data):
        dim.fit(data)
        data_transform = dim.transform(data)
        data_projected = data_transform.dot(dim.components_) + np.mean(data.values.ravel(), axis = 0)
        print((((data - data_projected) ** 2).mean()).mean())
        


    def reconstructionError(self):
        pca = PCA(n_components = self.pcaBest)
        self.calcError(pca, self.pcaData)
        
        ica = FastICA(n_components = self.icaBest, max_iter=1000)
        self.calcError(ica, self.icaData)

        rp = random_projection.GaussianRandomProjection(n_components = self.rpBest)
        self.calcError1(rp, self.rpData)

        
        
        fa = FactorAnalysis(n_components = self.faBest, max_iter=1000)
        self.calcError1(fa, self.faData)

    def loadData(self):
        if(self.dataset == "ford"):
            f = open("../data/fordReady.csv")
            self.data = pd.read_csv(f)
            self.dataX = self.data.ix[:, self.data.columns != "IsAlert"]
            self.dataY = self.data.ix[:, self.data.columns == "IsAlert"]
            f.close()
        else:
            f = open("../data/ottoReady.csv")
            self.data = pd.read_csv(f)
            self.dataX = self.data.ix[:, self.data.columns != "target"]
            self.dataY = self.data.ix[:, self.data.columns == "target"]
            f.close()

        

        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.dataX,self.dataY,test_size = 0.3, random_state = 0)
        self.trainX = self.trainX.reset_index(drop=True)
        self.trainY = self.trainY.reset_index(drop=True)

        self.pcaData = self.dataX
        self.icaData = self.dataX
        self.rpData = self.dataX
        self.faData = self.dataX

        self.pcaDataX = self.dataX
        self.icaDataX = self.dataX
        self.rpDataX = self.dataX
        self.faDataX = self.dataX

        self.pcaDataY = self.dataY
        self.icaDataY = self.dataY
        self.rpDataY = self.dataY
        self.faDataY = self.dataY

        print(self.data.columns)
        print(self.data.shape)

    def expLast(self, bow="kmeans"):
        def convert(a):
            temp = []
            for i in a:
                temp.append([i])
            return np.array(temp)
        def convert1(numClusters, A, clf="kmeans"):
            clf = KMeans(n_clusters= numClusters, init='k-means++')
            if(clf != "kmeans"):
                clf =GaussianMixture(n_components=numClusters)
            clf.fit(A)
            labels = convert((clf.predict(A)))
            return np.append(A, labels, axis=1)

        pca = PCA(n_components = self.pcaBest)
        pca.fit(self.pcaDataX)
        self.pcaDataX = pca.transform(self.pcaDataX)
        print(self.pcaDataX.shape)
        

        self.pcaDataX = convert1(self.pcaBest, self.pcaDataX, bow)
        print(self.pcaDataX.shape)

        self.pcaTrainX, self.pcaTestX, self.pcaTrainY, self.pcaTestY = train_test_split(self.pcaDataX,self.pcaDataY,test_size = 0.3, random_state = 0)
        

        ica = FastICA(n_components = self.icaBest, max_iter=1000)
        ica.fit(self.icaDataX)
        self.icaDataX = ica.transform(self.icaDataX)
        print(self.icaDataX.shape)

        self.icaDataX = convert1(self.icaBest, self.icaDataX, bow)

        self.icaTrainX, self.icaTestX, self.icaTrainY, self.icaTestY = train_test_split(self.icaDataX,self.icaDataY,test_size = 0.3, random_state = 0)
        print(self.icaDataX.shape)
        


        rp = random_projection.GaussianRandomProjection(n_components = self.rpBest)
        rp.fit(self.rpDataX)
        self.rpDataX = rp.transform(self.rpDataX)
        print(self.rpDataX.shape)

        self.rpDataX = convert1(self.rpBest, self.rpDataX, bow)

        self.rpTrainX, self.rpTestX, self.rpTrainY, self.rpTestY = train_test_split(self.rpDataX,self.rpDataY,test_size = 0.3, random_state = 0)
        print(self.rpDataX.shape)
        


        fa = FactorAnalysis(n_components = self.faBest, max_iter=1000)
        fa.fit(self.faDataX)
        self.faDataX = fa.transform(self.faDataX)
        print(self.faDataX.shape)

        self.faDataX = convert1(self.faBest, self.faDataX, bow)
        print(self.faDataX.shape)

        self.faTrainX, self.faTestX, self.faTrainY, self.faTestY = train_test_split(self.faDataX,self.faDataY,test_size = 0.3, random_state = 0)

        print(self.dataX.shape)
        self.dataX = convert1(14, self.dataX, bow)
        print(self.dataX.shape)

        normalResults = self.mlpc(self.trainX, self.trainY, self.testX, self.testY)
        pcaResults = self.mlpc(self.pcaTrainX, self.pcaTrainY, self.pcaTestX, self.pcaTestY)
        icaResults = self.mlpc(self.icaTrainX, self.icaTrainY, self.icaTestX, self.icaTestY)
        rpResults = self.mlpc(self.rpTrainX, self.rpTrainY, self.rpTestX, self.rpTestY)
        faResults = self.mlpc(self.faTrainX, self.faTrainY, self.faTestX, self.faTestY)
        print(normalResults)
        print(pcaResults)
        print(icaResults)
        print(rpResults)
        print(faResults)
        
    def mlpc(self, trainX, trainY, testX, testY, numNeurons = 15, numLayers = 1, activationFunction = "relu",):
        startTrainingTime = time.time()
        hidden_layer_sizes = tuple([numNeurons]*numLayers)
        mlpc = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
        mlpc.fit(trainX, trainY)
        endTrainingTime = time.time()
        trainingTime = endTrainingTime - startTrainingTime
        
        print("Time taken to train %s training samples is %s seconds" %(str(len(trainX)), str(trainingTime)))

        train_predict = mlpc.predict(trainX)
        test_predict = mlpc.predict(testX)

        trainAccuracy = accuracy_score(trainY, train_predict.round())
        # validationAccuracy = np.mean(cross_val_score(mlpc, trainX, trainY, cv=3))
        testAccuracy = accuracy_score(testY, test_predict.round())
        print("Training Accuracy %s" %(str(trainAccuracy)))
        # print("Validation Accuracy %s" %(str(validationAccuracy)))
        print("Testing Accuracy %s" %(str(testAccuracy)))
        print("=====================\n")
        print("\n")
        return [trainingTime, trainAccuracy, testAccuracy]

    def expMlpc(self):
        pca = PCA(n_components = self.pcaBest)
        pca.fit(self.pcaDataX)
        self.pcaDataX = pca.transform(self.pcaDataX)
        self.pcaTrainX, self.pcaTestX, self.pcaTrainY, self.pcaTestY = train_test_split(self.pcaDataX,self.pcaDataY,test_size = 0.3, random_state = 0)
        print(self.pcaTrainX.shape)

        ica = FastICA(n_components = self.icaBest, max_iter=1000)
        ica.fit(self.icaDataX)
        self.icaDataX = ica.transform(self.icaDataX)
        self.icaTrainX, self.icaTestX, self.icaTrainY, self.icaTestY = train_test_split(self.icaDataX,self.icaDataY,test_size = 0.3, random_state = 0)
        print(self.icaTrainX.shape)


        rp = random_projection.GaussianRandomProjection(n_components = self.rpBest)
        rp.fit(self.rpDataX)
        self.rpDataX = rp.transform(self.rpDataX)
        self.rpTrainX, self.rpTestX, self.rpTrainY, self.rpTestY = train_test_split(self.rpDataX,self.rpDataY,test_size = 0.3, random_state = 0)
        print(self.rpTrainX.shape)


        fa = FactorAnalysis(n_components = self.faBest, max_iter=1000)
        fa.fit(self.faDataX)
        self.faDataX = fa.transform(self.faDataX)
        self.faTrainX, self.faTestX, self.faTrainY, self.faTestY = train_test_split(self.faDataX,self.faDataY,test_size = 0.3, random_state = 0)
        print(self.faTrainX.shape)

        normalResults = self.mlpc(self.trainX, self.trainY, self.testX, self.testY)
        pcaResults = self.mlpc(self.pcaTrainX, self.pcaTrainY, self.pcaTestX, self.pcaTestY)
        icaResults = self.mlpc(self.icaTrainX, self.icaTrainY, self.icaTestX, self.icaTestY)
        rpResults = self.mlpc(self.rpTrainX, self.rpTrainY, self.rpTestX, self.rpTestY)
        faResults = self.mlpc(self.faTrainX, self.faTrainY, self.faTestX, self.faTestY)

        print(normalResults)
        print(pcaResults)
        print(icaResults)
        print(rpResults)
        print(faResults)

    def kMeansExp(self):
        self.trainTime = []
        self.pcaTrain = []
        self.icaTrain = []
        self.rpTrain = []
        self.faTrain = []
        self.silScore = []
        self.pcaSil = []
        self.icaSil = []
        self.rpSil = []
        self.faSil = []
        pca = PCA(n_components = self.pcaBest)
        pca.fit(self.pcaData)
        self.pcaData = pca.transform(self.pcaData)
        print(self.pcaData.shape)
        ica = FastICA(n_components = self.icaBest, max_iter=1000)
        ica.fit(self.icaData)
        self.icaData = ica.transform(self.icaData)
        rp = random_projection.GaussianRandomProjection(n_components = self.rpBest)
        rp.fit(self.rpData)
        self.rpData = rp.transform(self.rpData)
        fa = FactorAnalysis(n_components = self.faBest, max_iter=1000)
        fa.fit(self.faData)
        self.faData = fa.transform(self.faData)
        kClusters = list(np.arange(2,51,4))
        for num_classes in kClusters:
            print(num_classes)
            
            clf = KMeans(n_clusters= num_classes, init='k-means++')
            startTime = time.time()
            clf.fit(self.trainX)
            endTime = time.time()
            self.trainTime.append(endTime - startTime)
            labels = clf.labels_
            self.silScore.append(silhouette_score(self.trainX, labels, metric='euclidean'))

            clf = KMeans(n_clusters= num_classes, init='k-means++')
            startTime = time.time()
            clf.fit(self.pcaData)
            endTime = time.time()
            self.pcaTrain.append(endTime - startTime)
            labels = clf.labels_
            self.pcaSil.append(silhouette_score(self.pcaData, labels, metric='euclidean'))

            clf = KMeans(n_clusters= num_classes, init='k-means++')
            startTime = time.time()
            clf.fit(self.icaData)
            endTime = time.time()
            self.icaTrain.append(endTime - startTime)
            labels = clf.labels_
            self.icaSil.append(silhouette_score(self.icaData, labels, metric='euclidean'))

            clf = KMeans(n_clusters= num_classes, init='k-means++')
            startTime = time.time()
            clf.fit(self.rpData)
            endTime = time.time()
            self.rpTrain.append(endTime - startTime)
            labels = clf.labels_
            self.rpSil.append(silhouette_score(self.rpData, labels, metric='euclidean'))

            clf = KMeans(n_clusters= num_classes, init='k-means++')
            startTime = time.time()
            clf.fit(self.faData)
            endTime = time.time()
            self.faTrain.append(endTime - startTime)
            labels = clf.labels_
            self.faSil.append(silhouette_score(self.faData, labels, metric='euclidean'))

        print(self.silScore)
        print(self.pcaSil)
        print(self.icaSil)
        print(self.rpSil)
        print(self.faSil)
        print(self.trainTime)
        print(self.pcaTrain)
        print(self.icaTrain)
        print(self.rpTrain)
        print(self.faTrain)

        # self.silScore = [0.22544518755714213, 0.13385667831473738, 0.12686309661665648, 0.12701762356247498, 0.10329356940162236, 0.1090852652698505, 0.10552737228727375, 0.09496069538660969, 0.09719626916129331, 0.08833395452915888, 0.0901260114865964, 0.09561982702803452, 0.09180122133420889]
        # self.pcaSil = [0.22557345202617782, 0.13214420537579807, 0.12816032636623334, 0.12241112671546431, 0.10769180575724986, 0.09941079149227708, 0.08837245348641318, 0.09140616302973648, 0.0910460093383303, 0.08651859223030377, 0.09150225725986584, 0.08611154480807529, 0.08674987309150604]
        # self.icaSil = [0.22326582082797208, 0.09986706741182463, 0.09242798327730598, 0.06596936172719263, 0.044672545251517704, 0.04031730061440064, 0.041720072812167205, 0.03437669235769115, 0.029639360980545704, 0.024075246636092652, 0.023567700541853233, 0.02325035617217629, 0.021295696557617004]
        # self.rpSil = [0.17477074511047644, 0.07203302772808699, 0.10746874486661802, 0.0869008123890208, 0.06621899797110926, 0.07065615700238081, 0.05885707657237047, 0.04648369552281845, 0.05043925402351057, 0.036337097413448635, 0.04330250682557447, 0.04055120553423627, 0.039840831381554316]
        # self.faSil = [0.21383028575214746, 0.13582420374373835, 0.10978710865540973, 0.09066881531013209, 0.07739432098574484, 0.0954223498662894, 0.07072423043006205, 0.07466865323472115, 0.06916571513084652, 0.05425656104920422, 0.053373759381808904, 0.05477136790938987, 0.046735348706006416]
        # self.trainTime = [0.10041999816894531, 0.24351906776428223, 0.39322400093078613, 0.5157160758972168, 0.7793421745300293, 0.9012978076934814, 1.2145750522613525, 1.5887460708618164, 1.6785569190979004, 2.0500829219818115, 2.0654239654541016, 2.4878671169281006, 2.1492409706115723]
        # self.pcaTrain = [0.06250405311584473, 0.1833209991455078, 0.3063631057739258, 0.42432212829589844, 0.850085973739624, 0.9337520599365234, 0.9704890251159668, 1.6254260540008545, 1.6102569103240967, 1.9120128154754639, 1.7349450588226318, 2.161043882369995, 1.9693970680236816]
        # self.icaTrain = [0.09577107429504395, 0.17817211151123047, 0.25690484046936035, 0.38947010040283203, 0.616326093673706, 0.8578131198883057, 1.0042369365692139, 1.0739459991455078, 1.3683640956878662, 1.5702061653137207, 1.5731329917907715, 1.8525800704956055, 1.8427848815917969]
        # self.rpTrain = [0.12694811820983887, 0.32125020027160645, 0.5776569843292236, 0.7681779861450195, 0.9607009887695312, 1.248013973236084, 1.4433858394622803, 1.4887008666992188, 1.607158899307251, 2.227410078048706, 1.9074289798736572, 2.068984031677246, 2.2736399173736572]
        # self.faTrain = [0.08246803283691406, 0.17649507522583008, 0.2399449348449707, 0.39430809020996094, 0.5088560581207275, 0.7063210010528564, 0.8087069988250732, 1.0263988971710205, 1.2530300617218018, 1.4097480773925781, 1.504288911819458, 1.8240680694580078, 1.5019021034240723]
        # if(self.dataset == "otto"):
        #     self.silScore = [0.31531229941997935, 0.1770695243595119, 0.19800009500772855, 0.167351720523802, 0.1154252858528183, 0.12401660077438172, 0.12140921761941818, 0.12686760968286678, 0.11478481153725782, 0.12295616280986388, 0.1273514568056227, 0.10367434043180439, 0.12318605182531923]
        #     self.pcaSil = [0.3140503128141913, 0.1773871684406624, 0.17270337325635748, 0.1667758331877795, 0.10782396500808636, 0.1289423165138409, 0.12638411866715504, 0.12274230731641, 0.11252603723155033, 0.11065564976103939, 0.11025584533091139, 0.09489910835085587, 0.09985235069721334]
        #     self.icaSil = [0.24209355919470305, 0.19016035000795398, 0.1676446370778998, 0.14033790120946346, 0.16108372383431943, 0.1380871692570734, 0.14918115229750123, 0.13128596552965327, 0.12168000733710119, 0.12349332473006033, 0.1145226187830064, 0.09890568760330462, 0.06319438858463669]
        #     self.rpSil = [0.30577206214487457, 0.17211794653330575, 0.18772099489844998, 0.12858152232476253, 0.11361375440187685, 0.10531067179261579, 0.10846167644584664, 0.10240440410893646, 0.09353537956410676, 0.09412467053043733, 0.09969916304106122, 0.092695215908143, 0.09148584215320273]
        #     self.faSil = [0.24441919759594474, 0.1787270737956056, 0.2181766794756668, 0.17858922640767141, 0.17350831180558826, 0.16247050191305068, 0.15702708629229556, 0.15485801225644275, 0.1295756551051765, 0.10645806815867659, 0.12234478069236704, 0.10531379005789042, 0.08474113751836605]
        #     self.trainTime = [0.3212308883666992, 0.5400948524475098, 0.7268879413604736, 1.0735998153686523, 1.249701976776123, 1.426804780960083, 1.7800548076629639, 1.8528060913085938, 2.1698691844940186, 2.0919299125671387, 2.1053571701049805, 2.462557077407837, 2.542978048324585]
        #     self.pcaTrain = [0.16624784469604492, 0.3071861267089844, 0.37569594383239746, 0.5660111904144287, 0.6953039169311523, 0.8357110023498535, 1.0196070671081543, 1.4646620750427246, 1.7196869850158691, 1.4999921321868896, 1.537553071975708, 1.355332851409912, 2.137179136276245]
        #     self.icaTrain = [0.17263317108154297, 0.45412588119506836, 0.670647144317627, 0.6826670169830322, 0.7803261280059814, 0.8999981880187988, 0.9065568447113037, 1.1788480281829834, 1.2499349117279053, 1.2313120365142822, 1.3113040924072266, 1.6204321384429932, 1.6228270530700684]
        #     self.rpTrain = [0.1650688648223877, 0.41850805282592773, 0.6956040859222412, 0.8839321136474609, 0.8423540592193604, 1.3030691146850586, 1.2657239437103271, 1.6522178649902344, 1.9638760089874268, 1.5281641483306885, 1.6988651752471924, 1.8534560203552246, 2.1061198711395264]
        #     self.faTrain = [0.13854193687438965, 0.3004488945007324, 0.4537689685821533, 0.604604959487915, 0.640146017074585, 0.7561640739440918, 0.7801339626312256, 1.1896259784698486, 1.0693368911743164, 1.2788538932800293, 1.3081228733062744, 1.4440288543701172, 1.463705062866211]
        self.plotExp1(kClusters)

    def EMExp(self):
        self.trainTime = []
        self.pcaTrain = []
        self.icaTrain = []
        self.rpTrain = []
        self.faTrain = []
        self.bic = []
        self.pcaBic = []
        self.icaBic = []
        self.rpBic = []
        self.faBic = []
        pca = PCA(n_components = self.pcaBest)
        pca.fit(self.pcaData)
        self.pcaData = pca.transform(self.pcaData)
        print(self.pcaData.shape)
        ica = FastICA(n_components = self.icaBest, max_iter=1000)
        ica.fit(self.icaData)
        self.icaData = ica.transform(self.icaData)
        print(self.icaData.shape)
        rp = random_projection.GaussianRandomProjection(n_components = self.rpBest)
        rp.fit(self.rpData)
        self.rpData = rp.transform(self.rpData)
        print(self.rpData.shape)
        fa = FactorAnalysis(n_components = self.faBest, max_iter=1000)
        fa.fit(self.faData)
        self.faData = fa.transform(self.faData)
        print(self.faData.shape)
        kClusters = list(np.arange(2,51,4))
        for num_classes in kClusters:
            print(num_classes)
            
            
            clf = GaussianMixture(n_components=num_classes)
            startTime = time.time()
            clf.fit(self.dataX)
            endTime = time.time()
            self.trainTime.append(endTime - startTime)
            bic = clf.bic(self.dataX)
            self.bic.append(bic)

            clf = GaussianMixture(n_components=num_classes)
            startTime = time.time()
            clf.fit(self.pcaData)
            endTime = time.time()
            self.pcaTrain.append(endTime - startTime)
            bic = clf.bic(self.pcaData)
            self.pcaBic.append(bic)

            clf = GaussianMixture(n_components=num_classes)
            startTime = time.time()
            clf.fit(self.icaData)
            endTime = time.time()
            self.icaTrain.append(endTime - startTime)
            bic = clf.bic(self.icaData)
            self.icaBic.append(bic)

            clf = GaussianMixture(n_components=num_classes)
            startTime = time.time()
            clf.fit(self.rpData)
            endTime = time.time()
            self.rpTrain.append(endTime - startTime)
            bic = clf.bic(self.rpData)
            self.rpBic.append(bic)

            clf = GaussianMixture(n_components=num_classes)
            startTime = time.time()
            clf.fit(self.faData)
            endTime = time.time()
            self.faTrain.append(endTime - startTime)
            bic = clf.bic(self.faData)
            self.faBic.append(bic)

        print(self.bic)
        print(self.pcaBic)
        print(self.icaBic)
        print(self.rpBic)
        print(self.faBic)
        print(self.trainTime)
        print(self.pcaTrain)
        print(self.icaTrain)
        print(self.rpTrain)
        print(self.faTrain)
        self.plotExp2(kClusters)

    def plotExp2(self, x):
        plt.plot(x, self.bic,'-o',label="Normal",color="blue")
        plt.plot(x, self.pcaBic,'-o',label="PCA",color="red")
        plt.plot(x, self.icaBic,'-o',label="ICA",color="green")
        plt.plot(x, self.rpBic,'-o',label="RP",color="black")
        plt.plot(x, self.faBic,'-o',label="FA",color="orange")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("BIC")
        plt.title("BIC Vs #Clusters [" + self.dataset + "]")
        plt.xticks(x)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig("./exp2-" + self.dataset + "_scores.png")
        plt.close()

        plt.plot(x, self.trainTime,'-o',label="Normal",color="blue")
        plt.plot(x, self.pcaTrain,'-o',label="PCA",color="red")
        plt.plot(x, self.icaTrain,'-o',label="ICA",color="green")
        plt.plot(x, self.rpTrain,'-o',label="RP",color="black")
        plt.plot(x, self.faTrain,'-o',label="FA",color="orange")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Training Time in seconds")
        plt.title("Training Time Vs #Clusters [" + self.dataset + "]")
        plt.xticks(x)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig("./exp2-" + self.dataset + "_times.png")
        plt.close()

    def plotExp1(self, x):
        plt.plot(x, self.silScore,'-o',label="Normal",color="blue")
        plt.plot(x, self.pcaSil,'-o',label="PCA",color="red")
        plt.plot(x, self.icaSil,'-o',label="ICA",color="green")
        plt.plot(x, self.rpSil,'-o',label="RP",color="black")
        plt.plot(x, self.faSil,'-o',label="FA",color="orange")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score Vs #Clusters [" + self.dataset + "]")
        plt.xticks(x)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig("./exp1-" + self.dataset + "_scores.png")
        plt.close()

        plt.plot(x, self.trainTime,'-o',label="Normal",color="blue")
        plt.plot(x, self.pcaTrain,'-o',label="PCA",color="red")
        plt.plot(x, self.icaTrain,'-o',label="ICA",color="green")
        plt.plot(x, self.rpTrain,'-o',label="RP",color="black")
        plt.plot(x, self.faTrain,'-o',label="FA",color="orange")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Training Time in seconds")
        plt.title("Training Time Vs #Clusters [" + self.dataset + "]")
        plt.xticks(x)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig("./exp1-" + self.dataset + "_times.png")
        plt.close()

    def plotKnn(self, x, y, yLabel, title, fileName):
        plt.plot(x, y,'-o',label=yLabel,color="blue")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel(yLabel)
        plt.title(title)
        plt.xticks(x)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(fileName)
        plt.close()

    def plotKnnScores(self, x, fileName):
        plt.plot(x, self.silScore,'-o',label="Silhoutette Score",color="blue")
        plt.plot(x, self.compScore,'-o',label="Completeness Score",color="red")
        plt.plot(x, self.homoScore,'-o',label="Homogenity Score",color="green")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Score")
        plt.title("Performance Evaluation Scores for " + self.dataset)
        plt.xticks(x)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(fileName)
        plt.close()

    def plotEM(self, x, y, yLabel, title, fileName, xMin, yMin):
        plt.plot(x, y,'-o',label=yLabel,color="blue")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel(yLabel)
        plt.title(title)
        plt.xticks(x)
        plt.hlines(yMin, 0, 50, colors='red', linestyle = 'dashed', label='Lowest BIC at ' + str(xMin))
        plt.legend(loc='best')
        plt.savefig(fileName)
        plt.close()

    def plotClusterDistribution(self, k = 14, clusterAlgo="kmeans"):
        from collections import Counter, OrderedDict
        import operator
        clf = KMeans(n_clusters= k, init='k-means++')
        if(clusterAlgo != "kmeans"):
            clf = GaussianMixture(n_components=k)
        clf.fit(self.trainX)
        y_test_pred = clf.predict(self.testX)
        dist = Counter(y_test_pred)
        dist = OrderedDict(sorted(dist.items()))
        for i in range(k):
            try:
                temp = dist[i]
            except:
                dist[i] = 0
        print(dist)
        dist = OrderedDict(sorted(dist.items()))
        print(dist)

        xMax = max(dist.iteritems(), key=operator.itemgetter(1))[0]
        dist = dist.values()
        print(dist, len(dist))
        
        my_range = range(0,k)

        fig, ax = plt.subplots()
        ax.bar(my_range, dist, linewidth=2, color = 'blue')
        plt.axis('tight')
        plt.xlabel('Cluster labels')
        ax.set_ylabel('#Samples')
        plt.title(self.dataset + " dataset")
        plt.savefig("./" + clusterAlgo + "/" + self.dataset + "-distribution.png")
        plt.close()

        sil = silhouette_score(self.testX, y_test_pred, metric='euclidean')
        print("Sil Score", sil)
        print("AMI", adjusted_mutual_info_score(self.testY.values.ravel(), y_test_pred))

    def runKmeans(self, dimTransform=""):
        kClusters = list(np.arange(2,51,4))
        self.homoScore = []
        self.compScore = []
        self.silScore = []
        self.variance = []
        self.trainTime = []
        self.sse = []


        for num_classes in kClusters:
            print(num_classes)
            
            clf = KMeans(n_clusters= num_classes, init='k-means++')
            startTime = time.time()
            clf.fit(self.trainX)
            endTime = time.time()
            self.trainTime.append(endTime - startTime)
            

            y_test_pred = clf.predict(self.testX)
            # print(set(y_test_pred), set(self.testY.values.ravel()))

            self.sse.append(clf.inertia_)

            #Homogenity score on the test data
            homo = homogeneity_score(self.testY.values.ravel(), y_test_pred)
            self.homoScore.append(homo)
            
            #Completeness score
            comp = completeness_score(self.testY.values.ravel(), y_test_pred)
            self.compScore.append(comp)
            
            #Silhoutette score
            sil = silhouette_score(self.testX, y_test_pred, metric='euclidean')
            self.silScore.append(sil)

            #Variance explained by the cluster
            var = clf.score(self.testX)
            self.variance.append(var) 

        

        if(len(dimTransform) == 0):
            # self.plotKnn(kClusters, self.variance, "Variance explained by the cluster", "Variance Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./kmeans/" + self.dataset + "_kmeans-variance.png")
            # self.plotKnnScores(kClusters, self.silScore, "Silhoutette Score","Silhoutette Score Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./kmeans/" + self.dataset + "_kmeans-silhoutette.png")
            self.plotKnnScores(kClusters, "./kmeans/" + self.dataset + "_scores.png")
            self.plotKnn(kClusters, self.sse, "Sum of squared distance","Sum of squared distance Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./kmeans/" + self.dataset + "_kmeans-sse.png")
        else:
            # self.plotKnn(kClusters, self.variance, "Variance explained by the cluster", "Variance Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./kmeans-" + dimTransform + "/" + self.dataset + "_kmeans-variance.png")
            # self.plotKnn(kClusters, self.silScore, "Silhoutette Score","Silhoutette Score Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./kmeans-" + dimTransform + "/" + self.dataset + "_kmeans-silhoutette.png")
            # self.plotKnn(kClusters, self.homoScore, "Homogenity Score","Homogenity Score Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./kmeans-" + dimTransform + "/" + self.dataset + "_kmeans-Homogenity.png")
            # self.plotKnn(kClusters, self.compScore, "Completeness Score","Completeness Score Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./kmeans-" + dimTransform + "/" + self.dataset + "_kmeans-Completeness.png")
            self.plotKnn(kClusters, self.sse, "SSE","Sum of Squared Distance Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./kmeans-" + dimTransform + "/" + self.dataset + "_kmeans-sse.png")
            self.plotKnnScores(kClusters, "./kmeans-" + dimTransform + "/" + self.dataset + "_scores.png")
        self.plotClusterDistribution(k=14, clusterAlgo = "kmeans")

    def getMin(self):
        xMin = 2
        yMin = 10**20
        for i in range(len(self.bic)):
            print(self.bic[i], yMin)
            if(self.bic[i] < yMin):
                yMin = self.bic[i]
                xMin = 2 + (i*4)
        return xMin, yMin
        
    def runEM(self, no_iter=1000, dimTransform=""):
        kClusters = list(np.arange(2,51,4))

        self.aic = []
        self.bic = []
        self.avgLog = []

        self.homoScore = []
        self.compScore = []
        self.silScore = []
        self.trainTime = []

        for num_classes in kClusters:
            print(num_classes)

            clf = GaussianMixture(n_components=num_classes, max_iter=no_iter)
            
            startTime = time.time()
            clf.fit(self.dataX)
            endTime = time.time()
            self.trainTime.append(endTime - startTime)

            # y_test_pred = clf.predict(self.testX)

            # #Per sample average log likelihood
            # avg_log = clf.score(self.testX)
            # self.avgLog.append(avg_log)


            #AIC on the test data
            # aic = clf.aic(self.testX)
            # self.aic.append(aic)

            #BIC on the test data
            bic = clf.bic(self.dataX)
            self.bic.append(bic)

            #Homogenity score on the test data
            # homo = homogeneity_score(self.testY.values.ravel(), y_test_pred)
            # self.homoScore.append(homo)
            
            # #Completeness score
            # comp = completeness_score(self.testY.values.ravel(), y_test_pred)
            # self.compScore.append(comp)
            
            # #Silhoutette score
            # sil = silhouette_score(self.testX, y_test_pred, metric='euclidean')
            # self.silScore.append(sil)

        # if(self.dataset == "ford"):
        #     self.bic = [-60211.33300573715, -383508.4053738907, -500564.90536769, -600603.4440482798, -605419.323905269, -618838.9804070825, -633928.6550454632,  -659079.9802326616, -619892.6583978867, -649854.6782834452, -633342.3051436772, -615183.5195347971, -618635.6160579576]
        # else:
        #     self.bic = [1604399.6190365404, 580550.5978631979, 863766.5437320308, 472441.6861730201, 364100.97728643724, 339572.62780407455, 262035.33621335926, 666175.5445625619, 594844.273928284, 789039.4185639261, 925928.8948082055, 737437.1225275758, 958987.2927233682]
        xMin, yMin = self.getMin()
        print(self.bic, xMin, yMin)

        if(len(dimTransform) == 0):
            self.plotEM(kClusters, self.bic, "BIC Score", "BIC Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./em/" + self.dataset + "_bic.png", xMin, yMin)
        else:
            self.plotEM(kClusters, self.bic, "BIC Score", "BIC Vs #Clusters [" + self.datasetTitle[self.dataset] + "]", "./em-" + dimTransform + "/" + self.dataset + "_bic.png", xMin, yMin)

        if(self.dataset == "ford"):
            self.plotClusterDistribution(k=30, clusterAlgo="em")
        else:
            self.plotClusterDistribution(k=26, clusterAlgo="em")

    def plotDimRed(self, x, y, yLabel, title, fileName, xMark):
        plt.plot(x, y,'-.',label=yLabel,color="blue")
        plt.xlabel("Features")
        plt.ylabel(yLabel)
        plt.title(title)
        # plt.xticks(x)
        plt.vlines(xMark, 0, 1,linestyle=':', label='best n_components = %s' %(str(xMark)), linewidth = 2)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(fileName)
        plt.close()

    def cumulative(self, var):
        var1 = [0] * len(var)
        var1[0] = var[0]
        best_n = -1
        for i in range(1,len(var1)):
            var1[i] = var1[i-1] + var[i]
            if(var1[i] >= 0.85 and best_n == -1):
                best_n = i
        return var1, best_n

    def runPCA(self):
        numFeatures = 30
        if(self.dataset == "otto"):
            numFeatures = 93
        n_components = range(1, numFeatures + 1)
        print("Starting PCA")
        print("Dimensionality reduction")

        decisiontree = DecisionTreeClassifier(criterion = 'gini', max_depth = 15, min_samples_split = 5)
        pca = PCA()
        pipe = Pipeline(steps=[('pca', pca), ('decisionTree', decisiontree)])
        pca.fit(self.dataX)

        
        cumulativeVar, best_n = self.cumulative(pca.explained_variance_ratio_)
        self.plotDimRed(n_components, cumulativeVar, "Cumulative Explained Variance Ratio", "Cumulative Explained Variance Ratio Vs #Features [" + self.dataset + "]", "./pca/" + self.dataset + "-variance-line.png", best_n)


        fig, ax = plt.subplots()
        ax.bar(list(n_components), pca.explained_variance_ratio_, linewidth=2, color = 'blue')
        print(pca.explained_variance_ratio_)
        plt.axis('tight')
        plt.xlabel('n_components')
        ax.set_ylabel('Explained Variance Ratio')

        #Checking the accuracy for taking all combination of components
        
        gridSearch = GridSearchCV(pipe, dict(pca__n_components=n_components), cv = 3)
        gridSearch.fit(self.dataX, self.dataY)
        results = gridSearch.cv_results_
        # best_n = gridSearch.best_estimator_.named_steps['pca'].n_components
        ax1 = ax.twinx()

        #Plotting the accuracies and best component
        ax1.plot(results['mean_test_score'], linewidth = 2, color = 'red', label="CV score")
        ax1.set_ylabel('Mean Cross Validation Accuracy')
        ax1.axvline(best_n, linestyle=':', label='best n_components = %s' %(str(best_n)), linewidth = 2)

        plt.legend(prop=dict(size=12), loc="upper right")
        ax.plot()
        plt.title('CV score of DT & Variance explained for PCA [%s]'%self.dataset )
        plt.savefig("./pca/" + self.dataset + "_best-n_components.png")
        plt.close()

        # print("Clustering PCA")

        # pca_new = PCA(n_components = gridSearch.best_estimator_.named_steps['pca'].n_components)
        # pca_new.fit(self.trainX)
        # self.trainX = pca_new.transform(self.trainX)
        # self.testX = pca_new.transform(self.testX)
        # self.runKmeans(dimTransform="pca")
        # self.MLPC(numNeurons=15, numLayers = 1)

    def _calculate(self, X, ica_, n_components):
        
        components = ica_.components_
        ica_.components_ = components[:n_components]

        transformed = ica_.transform(X)
        ica_.components_ = components
     
        kurtosis = scipy.stats.kurtosis(transformed)

        return sorted(kurtosis, reverse = True) 

    def runICA(self):
        print("Starting ICA")
        print("Dimensionality reduction")
        numFeatures = 30
        if(self.dataset == "otto"):
            numFeatures = 93
        n_components = range(1, numFeatures + 1)

        decisiontree = DecisionTreeClassifier(criterion = 'gini', max_depth = 15, min_samples_split = 5)
        ica = FastICA(max_iter=1000)
        pipe = Pipeline(steps=[('ica', ica), ('decisionTree', decisiontree)])
        ica.fit(self.dataX)


        kurtosisScore = self._calculate(self.dataX, ica, numFeatures)
        cumulativeKurtosis, best_n = self.cumulative(kurtosisScore)
        self.plotDimRed(n_components, cumulativeKurtosis, "Cumulative kurtosis", "Cumulative kurtosis Vs #Features [" + self.dataset + "]", "./ica/" + self.dataset + "-variance-line.png", best_n)


        fig, ax = plt.subplots()
        ax.bar(n_components, kurtosisScore, linewidth=2, color = 'blue')
        plt.axis('tight')
        plt.xlabel('n_components')
        ax.set_ylabel('kurtosis')

        
        # gridSearch = GridSearchCV(pipe, dict(ica__n_components=n_components), cv = 3)
        # gridSearch.fit(self.dataX, self.dataY)
        # results = gridSearch.cv_results_ 
        # cvScores = results['mean_test_score']
        if(self.dataset == "ford"):
            cvScores = [0.652, 0.6753, 0.7062, 0.7682, 0.7811, 0.829, 0.8161, 0.8123, 0.815, 0.8251, 0.8191, 0.8143, 0.8099, 0.8083, 0.813, 0.817, 0.8052, 0.8141, 0.8092, 0.8122, 0.8125, 0.8084, 0.8166, 0.8219, 0.8243, 0.8224, 0.8282, 0.8181, 0.8166, 0.8205]
            best_n = 5
            yMax = 0.829
        else:
            cvScores = [0.5595, 0.5697, 0.566, 0.5725, 0.5917, 0.5988, 0.6017, 0.6115, 0.6293, 0.6225, 0.6319, 0.6331, 0.6566, 0.6413, 0.6591, 0.6676, 0.6647, 0.6721, 0.6623, 0.6733, 0.687, 0.6964, 0.6954, 0.7063, 0.7002, 0.6997, 0.705, 0.7015, 0.7026, 0.7012, 0.7067, 0.7063, 0.6987, 0.7037, 0.7037, 0.7002, 0.707, 0.7116, 0.7169, 0.6936, 0.7037, 0.7057, 0.7082, 0.698, 0.6886, 0.6889, 0.686, 0.6855, 0.6868, 0.6827, 0.6915, 0.683, 0.6868, 0.6893, 0.6761, 0.6769, 0.6794, 0.6728, 0.6644, 0.667, 0.6618, 0.6672, 0.658, 0.6689, 0.6656, 0.6505, 0.6621, 0.6521, 0.6459, 0.6432, 0.657, 0.6613, 0.6587, 0.6616, 0.6673, 0.6626, 0.6505, 0.6759, 0.6672, 0.6795, 0.6604, 0.6516, 0.6691, 0.6647, 0.6858, 0.677, 0.6592, 0.662, 0.678, 0.6516, 0.6424, 0.6424, 0.6404]
            best_n = 38
            yMax = 0.7169
        ax1 = ax.twinx()

        #Plotting the accuracies and best component
        ax1.plot(cvScores, linewidth = 2, color = 'red', label="CV score")
        ax1.set_ylabel('Mean Cross Validation Accuracy')

        ax1.axvline(best_n, linestyle=':', label='best n_components = %s' %(str(best_n)), linewidth = 2)
        # ax1.axhline(yMax, linestyle='-', color="green", linewidth = 2)

        plt.legend(prop=dict(size=12), loc="upper right")
        plt.title("Accuracy/kurtosis for ICA [" + self.dataset + "]")
        plt.tight_layout()
        plt.savefig("./ica/" + self.dataset + "_best-n_components.png")
        plt.close()

        # print("Clustering ICA")

        # ica_new = FastICA(n_components = gridSearch.best_estimator_.named_steps['ica'].n_components)
        # ica_new.fit(self.trainX)
        # self.trainX = ica_new.transform(self.trainX)
        # self.testX = ica_new.transform(self.testX)
        # self.runKmeans(dimTransform="ica")

    def runRP(self):
        def annot_max(x,y, ax=None):
            
            text= "x={:.3f}, y={:.3f}".format(x, y)
            if not ax:
                ax=plt.gca()
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.22)
            arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
            kw = dict(xycoords='data',textcoords="axes fraction",
                      arrowprops=arrowprops, bbox=bbox_props, ha="left", va="center")
            ax.annotate(text, xy=(x, y), xytext=(0.54,0.56), **kw)
        print("Starting RP")
        print("Dimensionality reduction")
        numFeatures = 30
        if(self.dataset == "otto"):
            numFeatures = 93
        n_components = range(1, numFeatures + 1)


        decisiontree = DecisionTreeClassifier(criterion = 'gini', max_depth = 15, min_samples_split = 5)
        rp = random_projection.GaussianRandomProjection(n_components = 30)
        pipe = Pipeline(steps=[('rp', rp), ('decisionTree', decisiontree)])
        rp.fit(self.dataX)

        fig, ax = plt.subplots()
        gridSearch = GridSearchCV(pipe, dict(rp__n_components=n_components), cv = 3)
        gridSearch.fit(self.dataX, self.dataY)
        results = gridSearch.cv_results_
        

        #Plotting the accuracies and best component
        plt.plot(n_components, results['mean_test_score'], linewidth = 2, color = 'red')
        plt.ylabel('Mean Cross Validation Accuracy over 10 iterations')
        plt.xlabel('n_components')
        plt.legend(prop=dict(size=12))
        plt.title("Accuracy VS #Features for RP [" + self.dataset + "]")
        plt.tight_layout()
        plt.show()
        best_n = float(input("Enter best_n"))
        best_y = float(input("Corresponding y value"))
        # best_n = gridSearch.best_estimator_.named_steps['rp'].n_components
        plt.plot(n_components, results['mean_test_score'], linewidth = 2, color = 'red')
        plt.ylabel('Mean Cross Validation Accuracy over 10 iterations')
        plt.xlabel('n_components')
        plt.legend(prop=dict(size=12))
        plt.title("Accuracy VS #Features for RP [" + self.dataset + "]")
        plt.tight_layout()
        ax.axvline(best_n, linestyle=':', label='best n_components = %s' %(str(best_n)), linewidth = 2)
        annot_max(best_n, best_y)

        
        plt.savefig("./rp/" + self.dataset + "_best-n_components.png")
        # plt.close()


        # #Reducing the dimensions with optimal number of components
        # print("Clustering RP")
        # rp_new = random_projection.GaussianRandomProjection(n_components = gridSearch.best_estimator_.named_steps['rp'].n_components)
        # rp_new.fit(self.trainX)
        # self.trainX = rp_new.transform(self.trainX)
        # self.testX = rp_new.transform(self.testX)
        # self.runKmeans(dimTransform="rp")

    def plotFAGraph(self, x, y, xMark):
        plt.plot(x, y,'-.',label="Eigen Value",color="blue")
        plt.xlabel("n_components")
        plt.ylabel("Eigen Value")
        plt.vlines(xMark, 0, max(y),linestyle=':', label='best n_components = %s' %(str(xMark)), linewidth = 2)
        plt.title("Eigen Value for varying n_components")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("./fa/" + self.dataset + "-eigen.png")
        plt.close()

    def runFA(self):
        print("Starting FA")
        print("Dimensionality reduction")
        numFeatures = 30
        if(self.dataset == "otto"):
            numFeatures = 93
        n_components = range(1, numFeatures + 1)

        decisiontree = DecisionTreeClassifier(criterion = 'gini', max_depth = 15, min_samples_split = 5)
        fa = FactorAnalysis(max_iter=1000)

        pipe = Pipeline(steps=[('fa', fa), ('decisionTree', decisiontree)])

        # Plot the fa spectrum
        fa.fit(self.dataX)
        X = fa.components_
        import numpy as np
        centered_matrix = X - X.mean(axis=1)[:, np.newaxis]
        cov = np.dot(centered_matrix, centered_matrix.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        best_n = 11
        if(self.dataset == "otto"):
            best_n = 30


        self.plotFAGraph(n_components, eigvals, best_n)

        fig, ax = plt.subplots()
        ax.bar(n_components, eigvals, linewidth=2, color = 'blue')
        plt.axis('tight')
        plt.xlabel('n_components')
        ax.set_ylabel('Eigen Values')

        

        gridSearch = GridSearchCV(pipe, dict(fa__n_components=n_components), cv = 3)
        gridSearch.fit(self.dataX, self.dataY)
        results = gridSearch.cv_results_
        ax1 = ax.twinx()

        #Plotting the accuracies and best component
        ax1.plot(results['mean_test_score'], linewidth = 2, color = 'red', label="CV score")
        ax1.set_ylabel('Mean Cross Validation Accuracy')
        ax1.axvline(best_n, linestyle=':', label='best n_components = %s'%(str(best_n)), linewidth = 2)

        plt.legend(prop=dict(size=12), loc="upper right")
        plt.title("Accuracy of DT and Eigen Values of Latent Variables [" + self.dataset + "]")
        plt.savefig("./fa/" + self.dataset + "_best-n_components.png")
        plt.close()

        # print("Clustering FA")

        # fa_new = FastICA(n_components = gridSearch.best_estimator_.named_steps['fa'].n_components)
        # fa_new.fit(self.trainX)
        # self.trainX = fa_new.transform(self.trainX)
        # self.testX = fa_new.transform(self.testX)
        # self.runKmeans(dimTransform="fa")    

if __name__ == "__main__":
    expObj = UnsupervisedLearning("ford")
    expObj.runExperiment()
    expObj = UnsupervisedLearning("otto")
    expObj.runExperiment()

