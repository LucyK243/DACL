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

from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

class NNConf:
    """
    Neural network configuration: architecture & hyperparameters
    """
    def __init__(self,
        input_features,             # the number of input features
        nn_arch,                    # neural network architecture, list, neurons per hidden layers
        num_epochs=100,             # the number of epochs to train neural network
        batch_size=32,              # batch size
        batch_norm=True,            # batch normalization layers enabled or disabled
        dropout=0.1,                # dropout layers enabled (> 0) or disabled (= 0)
        clipping_value=0,           # clipping enabled (> 0) or disabled
        lr=1e-3,                    # learning rate
        weight_decay=1e-3,          # weight decay
    ):
        self.input_features = input_features
        self.nn_arch = nn_arch
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.clipping_value = clipping_value
        self.lr = lr
        self.weight_decay = weight_decay
        
        
class TLConf:
    """
    Transfer learning configuration
    """
    def __init__(self,
        num_layers,       # the number of initial layers to which TL is applied 
        freeze=True       # freeze layers (True) or reinitialize from the previous model (False)
    ):
        self.num_layers = num_layers
        self.freeze = freeze


class NeuralNet(nn.Module):
    def __init__(self, 
        nn_conf: NNConf, 
        tl_conf: TLConf = None, 
        prev_model = None
    ):
        super(NeuralNet, self).__init__()

        self.nn_conf = nn_conf        
        num_layers = len(self.nn_conf.nn_arch)
        
        if num_layers > 0:
            # input layer
            if self.nn_conf.batch_norm:
                layers = [
                    nn.Linear(self.nn_conf.input_features, self.nn_conf.nn_arch[0]),
                    nn.BatchNorm1d(self.nn_conf.nn_arch[0]),
                    nn.ReLU()
                ]
            else:
                layers = [
                    nn.Linear(self.nn_conf.input_features, self.nn_conf.nn_arch[0]),
                    nn.ReLU()    
                ]  
                
            # middle layers
            for i in range(1, num_layers):
                current_layer = []
                if self.nn_conf.dropout > 0:
                    current_layer.append(nn.Dropout(self.nn_conf.dropout))
                
                current_layer.append(nn.Linear(self.nn_conf.nn_arch[i - 1], self.nn_conf.nn_arch[i]))
                
                if self.nn_conf.batch_norm:
                    current_layer.append(nn.BatchNorm1d(self.nn_conf.nn_arch[i]))
                
                current_layer.append(nn.ReLU())
                layers.extend(current_layer)
                                
            # final layer
            layers.append(nn.Linear(self.nn_conf.nn_arch[-1], 1))
        else:
            # NN without hidden layers
            layers = [nn.Linear(self.nn_conf.input_features, 1)] 
        
        self.model = nn.Sequential(*layers).to(DEVICE)
        self.num_epochs = self.nn_conf.num_epochs
        self.clipping_value = self.nn_conf.clipping_value
        self.batch_size = self.nn_conf.batch_size
        
        if tl_conf != None:
            self.setup_transfer_learning(tl_conf, prev_model)
        
        self.trainable_layers = self.determine_trainable_layers()
        # print(self.trainable_layers)
        
        # final step -- setup optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.nn_conf.lr, 
            weight_decay=self.nn_conf.weight_decay
        )
    
    def setup_transfer_learning(self, tl_conf, prev_model):
        # print("Setting up transfer learning")
        # print(tl_conf.num_layers, tl_conf.freeze)
        # print(type(prev_model)) 
        
        prev_model_layers = {}
        for name, layer in prev_model.named_children():
            prev_model_layers[name] = layer
        
        processed_layers = 0
        for name, layer in self.model.named_children():
            lin_layer = isinstance(layer, nn.Linear)
            stop = lin_layer and processed_layers == tl_conf.num_layers
            if stop:
                break
                
            if lin_layer:
                processed_layers += 1
                    
            bn_layer = isinstance(layer, nn.BatchNorm1d)
            
            # reuse previous model state
            layer.load_state_dict(prev_model_layers[name].state_dict())
            
            if lin_layer or bn_layer:    
                if tl_conf.freeze:
                    # print("Freezing", name)
                    for param in layer.parameters():
                        param.requires_grad = False   
    
    def determine_trainable_layers(self):
        tl = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                tl.append(name)
        return tl
    
    
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
    #return [3 * num_features // 4, num_features // 2, num_features // 4]
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
        nn_conf: NNConf,
        tl_conf: TLConf = None,
        num_iterations: int = 5, 
        ground_truth: DataTable=None, 
        verbose=False
    ):
        self.start_time = datetime.now()
        print("MVINN started", self.start_time)        
        self.verbose = verbose
        self.nn_conf = nn_conf
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
                    
        self.tl_conf = tl_conf
        self.tl_enabled = self.tl_conf != None
        print("Transfer learning enabled", self.tl_enabled)
                    
    
    def mvi_iteration(self, iteration_no, current_matrix):
        # do scaling
        scaler = StandardScaler()
        scaler.fit(current_matrix)
        data_matrix = scaler.transform(current_matrix)
        
        next_matrix = data_matrix.copy()
        
        num_features = data_matrix.shape[1]
        # input_features = num_features - 1
        
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
            if iteration_no > 0 and self.tl_enabled:
                nn = NeuralNet(self.nn_conf, self.tl_conf, self.NN[target_feature].model)
            else:
                nn = NeuralNet(self.nn_conf)
            
            loss_start, loss_end = nn.train(X, Y)

            # make predictions for missing values
            predicted_Y = nn.predict(inference_X)
            
            # populate next data matrix with prediction for current feature
            for i in range(len(predicted_Y)):
                row = missing_instances[i]
                next_matrix[row][target_feature] = predicted_Y[i][0]
            
            self.NN[target_feature] = nn
            
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
        # setup neural network dictionary
        self.NN = {}
        
        dm = self.start_matrix
        for i in range(self.num_iterations):
            dm = self.mvi_iteration(i, dm)
            print("Iteration", i, "finished", datetime.now())
            
        self.end_time = datetime.now()
        time_elapsed = self.end_time - self.start_time 
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
            
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
    from reproducibility import enable_reproducible_results
    enable_reproducible_results()
    
    dt = DataTable('data/sie-dataset1-mv.csv', columns_to_ignore=['Time'])
    gt = DataTable('data/sie-dataset1.csv', columns_to_ignore=['Time'])
    
    num_features = dt.data_matrix.shape[1]
    input_features = num_features - 1
    
    nn_conf = NNConf(input_features, default_NN_arch(input_features), batch_norm=False, batch_size=32, dropout=0.1)
    tl_conf = TLConf(2, freeze=False)
    
    mvinn = MVINN(dt, nn_conf, tl_conf=tl_conf, ground_truth=gt, num_iterations=5, verbose=False)
    mvinn.impute()
    