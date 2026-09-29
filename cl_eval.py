"""
PANDORA 

DACL, evaluation of classification models after missing class inference

author: lucy@dmi.uns.ac.rs
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree, svm, neighbors, naive_bayes, ensemble
from sklearn.model_selection import KFold
import numpy as np
import time

from imblearn.under_sampling import RandomUnderSampler

from datatable import DataTable

class SKLCModel:
    def __init__(self, name, classifier):
        self.classifier = classifier
        self.modelName = name

    def train(self, predictors, target):
        self.classifier.fit(predictors, target)
        
    def evaluate(self, predictors, target):
        predictions = self.classifier.predict(predictors)

        acc = accuracy_score(target, predictions)
        prec = precision_score(target, predictions, average="macro", zero_division=0)
        rec = recall_score(target, predictions, average="macro", zero_division=0)
        f1 = f1_score(target, predictions, average="macro", zero_division=0)

        return acc, prec, rec, f1
    


class SVM_SKLC(SKLCModel):
    def __init__(self, kernel="rbf"):
        super().__init__("SVM", svm.SVC(kernel=kernel))


class RF_SKLC(SKLCModel):
    def __init__(self, numEstimators=5):
        cm = ensemble.RandomForestClassifier(n_estimators=numEstimators)
        super().__init__("RF", cm)


class NB_SKLC(SKLCModel):
    def __init__(self):
        super().__init__("NB", naive_bayes.GaussianNB())


class KNN_SKLC(SKLCModel):
    def __init__(self, numNeighbors=5):
        cm = neighbors.KNeighborsClassifier(n_neighbors=numNeighbors)
        super().__init__("KNN", cm)


class DT_SKLC(SKLCModel):
    def __init__(self):
        super().__init__("DT", tree.DecisionTreeClassifier())


class KFoldCL:
    def __init__(self, predictors, target, numFolds=10):
        self.predictors = predictors
        self.target = target
        self.kf = KFold(n_splits=numFolds, shuffle=True, random_state=12)

    def eval(self, model):
        accL, precL, recL, f1L = [], [], [], []
        for trainSelection, testSelection in self.kf.split(self.predictors):
            trainPredictors = self.predictors[trainSelection]
            trainTarget = self.target[trainSelection]
            testPredictors = self.predictors[testSelection]
            testTarget = self.target[testSelection]

            model.train(trainPredictors, trainTarget)
            acc, prec, rec, f1 = model.evaluate(testPredictors, testTarget)
            accL.append(acc)
            precL.append(prec)
            recL.append(rec)
            f1L.append(f1)

        return np.mean(accL), np.mean(precL), np.mean(recL), np.mean(f1L)

    def evalMultiple(self, models):
        res = {}
        for m in models:
            start = time.time()
            acc, prec, rec, f1 = self.eval(m)
            end = time.time()
            totalTime = end - start
            res[m.modelName] = {
                "acc": acc,
                "prec": prec,
                "rec" : rec,
                "f1" : f1,
                "t": totalTime
            }

        return res

def evaluate_ds(predictors, targets, target_names, undersampling=False):
    best_per_target = []
    for target_name in target_names:
        print("Evaluating", target_name)
        target = targets[target_name].to_numpy()
        
        cmodels = [
            SVM_SKLC(), NB_SKLC(), KNN_SKLC(), DT_SKLC(), RF_SKLC()
        ]

        if undersampling:
            rus = RandomUnderSampler(random_state=42)
            predictors, target = rus.fit_resample(predictors, target)

        kfold = KFoldCL(predictors, target)
        res = kfold.evalMultiple(cmodels)
        
        print("CLASSIFIER,ACCURACY,PRECISION,RECALL,F1")
        max_f1 = 0
        max_f1_info = ""
        for cl in res:
            a, p, r, f1 = res[cl]['acc'], res[cl]['prec'], res[cl]['rec'], res[cl]['f1']
            info = cl + "," + str(round(a, 4)) + "," + str(round(p, 4)) + "," + str(round(r, 4)) + "," + str(round(f1, 4))
            print(info)
            if f1 > max_f1:
                max_f1 = f1
                max_f1_info = info

        print("----> BEST CLASSIFIER")
        print(max_f1_info)

        best_per_target.append((target_name, max_f1_info))

    return best_per_target   

def evaluate_dataset(data_table):
    predictors = data_table.data_matrix
    targets = data_table.target_vars_data
    target_names = list(targets.columns)
    
    return evaluate_ds(predictors, targets, target_names)

       
if __name__ == '__main__':
    from reproducibility import enable_reproducible_results
    enable_reproducible_results()
    
    #print("LOADING GROUND TRUTH")
    #ground_truth = DataTable('data/eab/eab-dataset2-labeled.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id', 'throughput (MB/s)', 'Latency(ms)'], verbose=False, sep=',')
    #ground_truth = DataTable('data/eab-1/eab1-full.csv', target_vars=['ANOMALY'], verbose=False, sep=',')
    #evaluate_dataset(ground_truth)

    datasets = [
        'data/eab-1/eab1-full-15_imp_LP.csv',
        'data/eab-1/eab1-full-15_imp_LS.csv',
        'data/eab-1/eab1-full-15_imp_NB.csv',
        'data/eab-1/eab1-full-15_imp_TABPFN.csv'
    ]

    imp_method = [
        "LP", "LS", "NB", "TABPFN"
    ]

    summary = []

    for i in range(len(imp_method)):
        print()
        print("EVALUATING CLASSIFIERS FOR", imp_method[i])
        dt = DataTable(datasets[i], target_vars=['ANOMALY'], verbose=False, sep=',')
        #dt = DataTable(datasets[i], target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id', 'throughput (MB/s)', 'Latency(ms)'], verbose=False, sep=',')
        best_cl = evaluate_dataset(dt)
        summary.append((imp_method[i], best_cl))

    print()
    print("SUMMARY")
    for s in summary:
        print(s)

