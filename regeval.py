"""
PANDORA 

DACL, evaluation of regression models after MVI

authors: {svc, lucy}@dmi.uns.ac.rs
"""

from datatable import DataTable

from sklearn import linear_model, svm, kernel_ridge
from sklearn import ensemble, neighbors
from sklearn.dummy import DummyRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import time


class SKLRModel:
    def __init__(self, regMethod, name):
        self.regMethod = regMethod
        self.modelName = name

    def train(self, predictors, target):
        self.regMethod.fit(predictors, target)
        
    def evaluate(self, predictors, target):
        predictions = self.regMethod.predict(predictors)
        mae = mean_absolute_error(target, predictions)
        mse = mean_squared_error(target, predictions)
        r2 = r2_score(target, predictions)
        return mae, mse, r2


class DummyReg(SKLRModel):
    def __init__(self):
        super().__init__(DummyRegressor(strategy="mean"), "Dummy")


class LinearReg_SKLR(SKLRModel):
    def __init__(self):
        super().__init__(linear_model.LinearRegression(), "Linreg")


class RidgeReg_SKLR(SKLRModel):
    def __init__(self):
        super().__init__(linear_model.RidgeCV(), "Ridge")


class LassoReg_SKLR(SKLRModel):
    def __init__(self):
        super().__init__(linear_model.LassoCV(), "Lasso")


class ElasticNetReg_SKLR(SKLRModel):
    def __init__(self):
        super().__init__(linear_model.ElasticNetCV(), "Elnet")


class KernelRidgeReg_SKLR(SKLRModel):
    def __init__(self):
        super().__init__(kernel_ridge.KernelRidge(), "KRidge")


class SGDReg_SKLR(SKLRModel):
    def __init__(self):
        super().__init__(linear_model.SGDRegressor(), "SGD")


class SVM_SKLR(SKLRModel):
    def __init__(self):
        super().__init__(svm.SVR(), "SVM")


class RF_SKLR(SKLRModel):
    def __init__(self, numEstimators=50):
        super().__init__(ensemble.RandomForestRegressor(n_estimators=numEstimators), "RF")


class KNN_SKLR(SKLRModel):
    def __init__(self, k=5):
        super().__init__(neighbors.KNeighborsRegressor(n_neighbors=k), "KNN")


class AdaBoost_SKLR(SKLRModel):
    def __init__(self, numEstimators=50):
        super().__init__(ensemble.AdaBoostRegressor(n_estimators=numEstimators), "ADA")


class KFoldReg:
    def __init__(self, predictors, target, numFolds=10):
        self.predictors = predictors
        self.target = target
        self.kf = KFold(n_splits=numFolds, shuffle=True, random_state=12)

    def eval(self, model):
        maeL, mseL, r2L = [], [], []
        for trainSelection, testSelection in self.kf.split(self.predictors):
            trainPredictors = self.predictors[trainSelection]
            trainTarget = self.target[trainSelection]
            testPredictors = self.predictors[testSelection]
            testTarget = self.target[testSelection]

            model.train(trainPredictors, trainTarget)
            mae, mse, r2 = model.evaluate(testPredictors, testTarget)
            maeL.append(mae)
            mseL.append(mse)
            r2L.append(r2)

        return np.mean(maeL), np.mean(mseL), np.mean(r2L)

    def evalMultiple(self, models):
        res = {}
        for m in models:
            start = time.time()
            mae, mse, r2 = self.eval(m)
            end = time.time()
            totalTime = end - start
            res[m.modelName] = {
                "mae": mae,
                "mse": mse,
                "r2" : r2,
                "t": totalTime
            }
        return res

def is_const(arr):
    f = arr[0]
    for e in arr:
        if e != f:
            return False
    return True

def evalreg(data, rmodels, target_feature):
    #num_features = data.data_matrix.shape[1]
    #print(num_features)
    #target_feature = 1
    print("Target feature", data.column_names[target_feature])
    Y = data.data_matrix[:, target_feature]
    X = np.delete(data.data_matrix, target_feature, axis=1)
    
    if is_const(Y):
        print("Constant Y skipped")
        return
    
    print(X.shape)
    print(Y.shape)

    kfcv = KFoldReg(X, Y)
    res_l = kfcv.evalMultiple(rmodels)
        
    minmae = float('inf')
    minm = None

    for m in rmodels:
        mn = m.modelName
        print(mn + "," + str(round(res_l[mn]["mae"], 4)) + "," +  str(round(res_l[mn]["r2"], 4)) + "," + str(round(res_l[mn]["mse"], 4)))
        mae = res_l[mn]["mae"]
        if mae < minmae:
            minmae = mae
            minr2 = res_l[mn]["r2"]
            minm = mn

    impDummy = 100.0 * res_l["Dummy"]["mae"] / minmae - 100.0
    print("The best model =", minm, ",", minmae, ", improvement(dummy) =", round(impDummy, 2), "%")

    return minm, minmae, minr2

def evaluateRegModels(data, target_feature):
    rmodels = [
        DummyReg(),
        LinearReg_SKLR(),
        RidgeReg_SKLR(),
        LassoReg_SKLR(),
        ElasticNetReg_SKLR(),
        SVM_SKLR(),
        KNN_SKLR(),
        RF_SKLR(),
    ]

    return evalreg(data, rmodels, target_feature)

def evaluateForAllFeatures(data):
    num_features = data.data_matrix.shape[1]
    final_rep = []
    print("MODEL,MAE,R2,MSE")
    for i in range(num_features):
        res = evaluateRegModels(data, i)
        if res != None:
            tf = data.column_names[i]
            s = tf + "," + res[0] + "," + str(round(res[1], 4)) + "," + str(round(res[2], 4))
            final_rep.append(s)
           
    print("\n\n") 
    print("-- FINAL REPORT")
    print("FEATURE,REGRESSOR,MAE,R2")
    for s in final_rep:
        print(s)

def evaluateForTargetFeatures(data, target_features):
    num_features = data.data_matrix.shape[1]
    final_rep = []
    for feature in target_features:
        i = None
        for col in range(num_features):
            if data.column_names[col] == feature:
                i = col
                break
        res = evaluateRegModels(data, i)
        if res != None:
            tf = data.column_names[i]
            s = tf + "," + res[0] + "," + str(round(res[1], 4)) + "," + str(round(res[2], 4))
            final_rep.append(s)
           
    print("\n\n") 
    print("-- FINAL REPORT")
    print("FEATURE,REGRESSOR,MAE,R2")
    for s in final_rep:
        print(s)

def tabPFNeval_HALCOR(data_file):
    import pandas as pd
    from tabpfn import TabPFNRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_file, sep=",")
    columns_to_drop = ['Energy', 'Timestamp']
    X = df.drop(columns=columns_to_drop)
    y = df['Energy']
    # t = df['Timestamp']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    X['Year'] = df['Timestamp'].dt.year
    X['Month'] = df['Timestamp'].dt.month
    X['Day'] = df['Timestamp'].dt.day
    X['Hour'] = df['Timestamp'].dt.hour
    X['Minute'] = df['Timestamp'].dt.minute

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=128, random_state=42
    )
    #print("Dataset:", X_train.shape, X_test.shape)
    #print("Dataset:", y_train.shape, y_test.shape)

    tabpfn_regressor = TabPFNRegressor(random_state=42)
    tabpfn_regressor.fit(X_train, y_train)
    y_pred = tabpfn_regressor.predict(X_test)

    score = r2_score(y_test, y_pred)
    print(f"The R2 score:", "{:.4f}".format(score))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"TabPFN RMSE: {rmse:.4f}")


if __name__ == '__main__':
    from reproducibility import enable_reproducible_results
    enable_reproducible_results()
    
    """
    dt = DataTable('data_for_ml/25/sie-dataset1-nntl-mv25.csv', columns_to_ignore=['Time'])
    evaluateForAllFeatures(dt)
    """

    # halcor
    
    print("ORIGINAL DATA")
    dt = DataTable('data/halcor/halcor1.csv', columns_to_ignore=['Timestamp'])
    evaluateForTargetFeatures(dt, ['Energy'])

    print("SIMPLE")
    dt = DataTable('data/halcor/halcor1-25-simple.csv', columns_to_ignore=['Timestamp'])
    evaluateForTargetFeatures(dt, ['Energy'])

    print("NN")
    dt = DataTable('data/halcor/halcor1-25-nn.csv', columns_to_ignore=['Timestamp'])
    evaluateForTargetFeatures(dt, ['Energy'])

    print("NNTL")
    dt = DataTable('data/halcor/halcor1-25-nntl.csv', columns_to_ignore=['Timestamp'])
    evaluateForTargetFeatures(dt, ['Energy'])
    
    
    # tabfpn
    print("\n\n\n")
    print("ORIGINAL")
    tabPFNeval_HALCOR('data/halcor/halcor1.csv')
    
    print("\n\nSIMPLE")
    tabPFNeval_HALCOR('data/halcor/halcor1-25-simple.csv')

    print("\n\nNN")
    tabPFNeval_HALCOR('data/halcor/halcor1-25-nn.csv')

    print("\n\nNNTL")
    tabPFNeval_HALCOR('data/halcor/halcor1-25-nntl.csv')
