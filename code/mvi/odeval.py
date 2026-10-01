"""
PANDORA 

DACL, evaluation of MVI based on outlier detection

authors: {svc, lucy}@dmi.uns.ac.rs
"""

import sys
sys.path.append('..')
from datatable import DataTable

from scipy.stats import kendalltau

#from pyod.models import knn, pca, hbos, cblof, lscp, auto_encoder, abod

from pyod.models import auto_encoder, ae1svm, cblof, deep_svdd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

class Base_PyOD:
    def __init__(self, od_method, od_method_name, outlier_thr=0.9):
        self.od_method = od_method
        self.od_method.verbose = False
        self.od_method_name = od_method_name
        self.outlier_thr = outlier_thr
        
    def fit(self, data_table):
        self.od_method.fit(data_table.data_matrix)
        scores = self.od_method.decision_scores_
        
        # normalization of scores to [0, 1]
        max_score = max(scores)
        self.outlier_scores = [s / max_score for s in scores]
        
        # compute the distribution of scores (10 bins)
        self.distr = [0] * 10
        for s in self.outlier_scores:
            bin = int(s * 10)
            if bin == 10:
                bin = 9
            
            self.distr[bin] += 1
            
        # form outlier lables
        self.num_outliers = 0
        self.outlier_labels = [0] * len(self.outlier_scores)
        for i in range(len(self.outlier_scores)):
            if self.outlier_scores[i] > self.outlier_thr:
                self.outlier_labels[i] = 1
                self.num_outliers += 1
        
    def get_outlier_scores(self):
        return self.outlier_scores
    
    def get_outlier_labels(self):
        return self.outlier_labels
    
    def get_num_outliers(self):
        return self.num_outliers
    
    def info(self):
        print("Outliers detected by", self.od_method_name)
        print("#outliers", self.num_outliers)
        for i in range(len(self.distr)):
            low = i * 0.1
            high = low + 0.1
            frac = self.distr[i] / len(self.outlier_scores)
            print(round(low, 1), round(high, 1), "--", self.distr[i], "--", round(frac, 4))
            
            
class AE_PyOD(Base_PyOD):
    def __init__(self):
        #od_method = auto_encoder.AutoEncoder(epoch_num=100, hidden_neuron_list=[32], batch_norm=False, verbose=0)
        #od_method = deep_svdd.DeepSVDD(47, epochs=200, hidden_neurons=[32, 16], verbose=False)
        od_method = ae1svm.AE1SVM(epochs=50, hidden_neurons=[32])
        super().__init__(od_method, "AE")
        
        
class KMeans_PyOD(Base_PyOD):
    def __init__(self):
        od_method = cblof.CBLOF()
        super().__init__(od_method, "KMeans")


def compare_outlier_scores(s1, s2):
    return kendalltau(s1, s2).statistic

def compare_outlier_labels(l1, l2):
    # return accuracy_score(l1, l2)
    p = precision_score(l1, l2, pos_label=1)
    r = recall_score(l1, l2, pos_label=1)
    if p == 0 and r == 0:
        return 0
    
    f1 = 2 * p * r / (p + r)
    return f1
    #return balanced_accuracy_score(l1, l2)

def outlier_report(data_path):
    dt = DataTable(data_path)
    print("Detecting outliers...")
    ae = AE_PyOD()
    ae.fit(dt)
    n = ae.get_num_outliers()
    print(n, " outliers detected")

### EXPERIMENTS

def compute_table_HALCOR():
    dt_base = DataTable('data/halcor/halcor1.csv', columns_to_ignore=['Timestamp'])
    ae = AE_PyOD()
    ae.fit(dt_base)
    ae.info()
    s_ae_base = ae.get_outlier_scores()
    l_ae_base = ae.get_outlier_labels()

    imputed = [
        ['data/halcor/halcor1-05-simple.csv', 'data/halcor/halcor1-05-nn.csv', 'data/halcor/halcor1-05-nntl.csv'],
        ['data/halcor/halcor1-10-simple.csv', 'data/halcor/halcor1-10-nn.csv', 'data/halcor/halcor1-10-nntl.csv'],
        ['data/halcor/halcor1-15-simple.csv', 'data/halcor/halcor1-15-nn.csv', 'data/halcor/halcor1-15-nntl.csv'],
        ['data/halcor/halcor1-20-simple.csv', 'data/halcor/halcor1-20-nn.csv', 'data/halcor/halcor1-20-nntl.csv'],
        ['data/halcor/halcor1-25-simple.csv', 'data/halcor/halcor1-25-nn.csv', 'data/halcor/halcor1-25-nntl.csv']
    ]

    print("SIMPLE,NN,NNTL")
    for imp in imputed:
        dt = DataTable(imp[0], columns_to_ignore=['Timestamp'])
        ae = AE_PyOD()
        ae.fit(dt)
        # ae.info()
        s = ae.get_outlier_scores()
        l = ae.get_outlier_labels()
        n1 = ae.get_num_outliers()
        c1 = compare_outlier_scores(s_ae_base, s)
        a1 = compare_outlier_labels(l_ae_base, l) 
        
        dt = DataTable(imp[1], columns_to_ignore=['Timestamp'])
        ae = AE_PyOD()
        ae.fit(dt)
        # ae.info()
        s = ae.get_outlier_scores()
        l = ae.get_outlier_labels()
        n2 = ae.get_num_outliers()
        c2 = compare_outlier_scores(s_ae_base, s)
        a2 = compare_outlier_labels(l_ae_base, l) 
        
        dt = DataTable(imp[2], columns_to_ignore=['Timestamp'])
        ae = AE_PyOD()
        ae.fit(dt)
        # ae.info()
        s = ae.get_outlier_scores()
        l = ae.get_outlier_labels()
        n3 = ae.get_num_outliers()
        c3 = compare_outlier_scores(s_ae_base, s)
        a3 = compare_outlier_labels(l_ae_base, l) 
        
        print(str(n1) + "," + str(n2) + "," + str(n3) + "," + str(c1) + "," + str(c2) + "," + str(c3) + "," + str(a1) + "," + str(a2) + "," + str(a3))
        a3 = compare_outlier_labels(l_ae_base, l) 

def compute_table_SIE():
    dt_base = DataTable('data_for_ml/sie-dataset1.csv', columns_to_ignore=['Time'])
    ae = AE_PyOD()
    ae.fit(dt_base)
    ae.info()
    s_ae_base = ae.get_outlier_scores()
    l_ae_base = ae.get_outlier_labels()
    
    imputed = [
        ['data_for_ml/5/sie-dataset-1-5-simple.csv', 'data_for_ml/5/sie-dataset1-nn-mv5.csv', 'data_for_ml/5/sie-dataset1-nntl-mv5.csv'],
        ['data_for_ml/sie-dataset1-simple.csv', 'data_for_ml/sie-dataset1-nn.csv', 'data_for_ml/sie-dataset1-nntl.csv'],
        ['data_for_ml/15/sie-dataset-1-15-simple.csv', 'data_for_ml/15/sie-dataset1-nn-mv15.csv', 'data_for_ml/15/sie-dataset1-nntl-mv15.csv'],
        ['data_for_ml/20/sie-dataset-1-20-simple.csv', 'data_for_ml/20/sie-dataset1-nn-mv20.csv', 'data_for_ml/20/sie-dataset1-nntl-mv20.csv'],
        ['data_for_ml/25/sie-dataset-1-25-simple.csv', 'data_for_ml/25/sie-dataset1-nn-mv25.csv', 'data_for_ml/25/sie-dataset1-nntl-mv25.csv'],
    ]
    
    print("SIMPLE,NN,NNTL")
    for imp in imputed:
        dt = DataTable(imp[0], columns_to_ignore=['Time'])
        ae = AE_PyOD()
        ae.fit(dt)
        # ae.info()
        s = ae.get_outlier_scores()
        l = ae.get_outlier_labels()
        n1 = ae.get_num_outliers()
        c1 = compare_outlier_scores(s_ae_base, s)
        a1 = compare_outlier_labels(l_ae_base, l) 
        
        dt = DataTable(imp[1], columns_to_ignore=['Time'])
        ae = AE_PyOD()
        ae.fit(dt)
        # ae.info()
        s = ae.get_outlier_scores()
        l = ae.get_outlier_labels()
        n2 = ae.get_num_outliers()
        c2 = compare_outlier_scores(s_ae_base, s)
        a2 = compare_outlier_labels(l_ae_base, l) 
        
        dt = DataTable(imp[2], columns_to_ignore=['Time'])
        ae = AE_PyOD()
        ae.fit(dt)
        # ae.info()
        s = ae.get_outlier_scores()
        l = ae.get_outlier_labels()
        n3 = ae.get_num_outliers()
        c3 = compare_outlier_scores(s_ae_base, s)
        a3 = compare_outlier_labels(l_ae_base, l) 
        
        print(str(n1) + "," + str(n2) + "," + str(n3) + "," + str(c1) + "," + str(c2) + "," + str(c3) + "," + str(a1) + "," + str(a2) + "," + str(a3))
        a3 = compare_outlier_labels(l_ae_base, l) 
    

if __name__ == '__main__':
    from reproducibility import enable_reproducible_results
    enable_reproducible_results()
    
    compute_table_HALCOR()
    
    #dt = DataTable('data_for_ml/sie-dataset1.csv', columns_to_ignore=['Time'])
    #print('Data shape', dt.data_matrix.shape)
    
    """
    ae = AE_PyOD()
    ae.fit(dt)
    ae.info()
    s_ae_base = ae.get_outlier_scores()

    km = KMeans_PyOD()
    km.fit(dt)
    km.info()
    s_km_base = km.get_outlier_scores()
    
    print("--> ", compare_outlier_scores(s_ae_base, s_km_base))
    
    dt = DataTable('data_for_ml/5/sie-dataset1-nntl-mv5.csv', columns_to_ignore=['Time'])
    ae = AE_PyOD()
    ae.fit(dt)
    ae.info()
    s_ae = ae.get_outlier_scores()

    km = KMeans_PyOD()
    km.fit(dt)
    km.info()    
    s_km = km.get_outlier_scores()
    
    print("--> ", compare_outlier_scores(s_ae, s_km))
    
    print("###")
    print("AE comparison", compare_outlier_scores(s_ae_base, s_ae))
    print("KM comparison", compare_outlier_scores(s_km_base, s_km))
    """