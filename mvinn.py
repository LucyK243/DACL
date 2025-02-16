"""
PANDORA 

DACL, missing value inference based on neural networks and transfer learning

authors: {svc, lucy}@dmi.uns.ac.rs
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from datatable import DataTable

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


def enable_reproducible_results(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # random.seed(seed)



class NeuralNet(nn.Module):
    def __init__(self, 
        input_features, 
        nn_arch, 
        num_epochs=100,
        batch_size=32,
        batch_norm=True,
        dropout=0,                   # default is 0.1 when using dropout
        clipping_value=0,            # default is 1 when clipping gradients 
        lr=1e-3,
        weight_decay=1e-3
    ):
        super(NeuralNet, self).__init__()
        
        num_layers = len(nn_arch)
        
        if num_layers > 0:
            # input layer
            if batch_norm:
                layers = [
                    nn.Linear(input_features, nn_arch[0]),
                    nn.BatchNorm1d(nn_arch[0]),
                    nn.ReLU()
                ]
            else:
                layers = [
                    nn.Linear(input_features, nn_arch[0]),
                    nn.ReLU()    
                ]  
                
            # middle layers
            for i in range(1, num_layers):
                current_layer = []
                if dropout > 0:
                    current_layer.append(nn.Dropout(dropout))
                
                current_layer.append(nn.Linear(nn_arch[i - 1], nn_arch[i]))
                
                if batch_norm:
                    current_layer.append(nn.BatchNorm1d(nn_arch[i]))
                
                current_layer.append(nn.ReLU())
                layers.extend(current_layer)
                                
            # final layer
            layers.append(nn.Linear(nn_arch[-1], 1))
        else:
            # NN without hidden layers
            layers = [nn.Linear(input_features, 1)] 
        
        self.model = nn.Sequential(*layers).to(DEVICE)
        
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        self.num_epochs = num_epochs
        self.clipping_value = clipping_value
        self.batch_size = batch_size
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
    
    
    def train(self, X, Y):
        _X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        _Y = torch.tensor(Y, dtype=torch.float32).to(DEVICE)
        
        dataset = TensorDataset(_X, _Y)
        loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=False)
        
        loss = nn.MSELoss()
        final_tl_loss, first_tl_loss = None, None
        
        for i in range(self.num_epochs):
            train_loss = []
            for batch_ndx, sample in enumerate(loader):
                self.optimizer.zero_grad()
                X_next, y_next = sample
                preds = self.forward(X_next).squeeze()
                batch_loss = loss(preds, y_next)
                batch_loss.backward()
                
                if self.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
        
                self.optimizer.step()
                train_loss.append(batch_loss.detach())
            
            train_loss = torch.Tensor(train_loss).to(DEVICE)
            if i == 0:
                first_tl_loss = torch.mean(train_loss)
            else:
                final_tl_loss = torch.mean(train_loss)
        
        return first_tl_loss, final_tl_loss   
     
     
    def predict(self, X):
        with torch.no_grad():
            _X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            return self.model(_X).detach().cpu().numpy()        
    

def default_NN_arch(num_features):
    return [num_features // 2, num_features // 4]


def is_const(arr):
    f = arr[0]
    for e in arr:
        if e != f:
            return False
    return True 
        

class MVINN:
    def __init__(self, 
        data_table: DataTable,
        num_iterations: int = 5, 
        ground_truth: DataTable=None, 
        verbose=False
    ):
        enable_reproducible_results()
        
        self.verbose = verbose
        self.num_iterations = num_iterations
        self.ground_truth = ground_truth
        self.data_table = data_table
        self.dimX, self.dimY = self.data_table.data_matrix.shape
        
        # do simple imputer
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(self.data_table.data_matrix)
        self.start_matrix = imp_mean.transform(self.data_table.data_matrix)
        
        # get missing value dictionary (what is missing in data table)
        self.mv_dict = self.data_table.missing_values_dict
        
        if ground_truth != None:
            self.start_mae, self.start_r2 = ground_truth.diff_data_matrix(self.start_matrix)
        
        if self.verbose:
            print("Initial simple imputation... DONE")
            if self.ground_truth != None:
                for cl in self.start_mae:
                    print(cl, self.start_mae[cl], self.start_r2[cl])
    
    
    def mvi_iteration(self, current_matrix):
        # do scaling
        scaler = StandardScaler()
        scaler.fit(current_matrix)
        data_matrix = scaler.transform(current_matrix)
        
        next_matrix = data_matrix.copy()
        
        num_features = data_matrix.shape[1]
        input_features = num_features - 1
        
        for target_feature in range(num_features):
        # for target_feature in range(1):
            fname, missing_instances = self.mv_dict[target_feature]
            
            # form training data
            training_data = np.delete(data_matrix, missing_instances, axis=0)
            Y = training_data[:, target_feature]
            X = np.delete(training_data, target_feature, axis=1)
            
            if is_const(Y):
                # print("Constant Y", fname, "skipped")
                continue
            
            # form inference data
            inference_data = np.take(data_matrix, missing_instances, axis=0)
            inference_X = np.delete(inference_data, target_feature, axis=1)
            
            # train neural network
            nn = NeuralNet(input_features, default_NN_arch(input_features), dropout=0.1)
            loss_start, loss_end = nn.train(X, Y)

            # make predictions for missing values
            predicted_Y = nn.predict(inference_X)
            
            # populate next data matrix with prediction for current feature
            for i in range(len(predicted_Y)):
                row = missing_instances[i]
                next_matrix[row][target_feature] = predicted_Y[i][0]
            
            if self.verbose:
                print()
                print("Finished training neural network for feature", target_feature, fname)
                print("Missing instances", len(missing_instances))
                #print("Training data shapes", X.shape, Y.shape)
                #print("Inference data shape", inference_X.shape)
                print("Loss start", loss_start.item(), "Loss end", loss_end.item())
                # print("Neural network arch", nn)
                
                
        
        # inverse rescaling
        next_matrix = scaler.inverse_transform(next_matrix)
        
        if self.verbose:
            if self.ground_truth != None:
                mae, r2 = self.ground_truth.diff_data_matrix(next_matrix)
                for cl in mae:
                    print(cl, mae[cl], r2[cl])
        
        return next_matrix
    
    
    def impute(self):
        dm = self.start_matrix
        for i in range(self.num_iterations):
            dm = self.mvi_iteration(dm)
            
        if self.ground_truth != None:
            self.final_mae, self.final_r2 = self.ground_truth.diff_data_matrix(dm)
            for cl in self.final_mae:
                print()
                print(cl)
                print("MAE", self.start_mae[cl], self.final_mae[cl])
                print("R^2", self.start_r2[cl], self.final_r2[cl])
            
                if self.start_mae[cl] == 0.0:
                    continue
            
                if self.final_mae[cl] < self.start_mae[cl]:
                    print("BETTER")
                    
                impr = (self.start_mae[cl] - self.final_mae[cl]) / self.start_mae[cl]     
                impr *= 100
                print(impr)     
         
                
            
if __name__ == '__main__':
    dt = DataTable('data/sie-dataset1-mv.csv', columns_to_ignore=['Time'])
    gt = DataTable('data/sie-dataset1.csv', columns_to_ignore=['Time'])
    
    mvinn = MVINN(dt, ground_truth=gt, verbose=False)
    mvinn.impute()
    