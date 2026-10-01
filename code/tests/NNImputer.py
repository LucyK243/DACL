import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import pandas as pd


class EABDataset(torch.utils.data.Dataset):
  '''
  Prepare the Boston dataset for regression
  '''

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]


class NNImputer(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #98 columns in total, 96 usable (-2 on bag_file and timestamp)
        self.layers = nn.Sequential(
            nn.Linear(92, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.layers(x)
    

if __name__ == '__main__':
    dataset = pd.read_csv('../data/eab_sample_zero_knn5.csv')
    columns_to_ignore=['timestamp', 'bag_file']
    for column in columns_to_ignore:
        dataset.drop(column, axis=1, inplace=True)

    target_columns = ['throughput_DL', 'throughput_UL', 'delay_mean_DL', 'delay_mean_UL']
    all_columns = dataset.columns.tolist()
    remaining_columns = [col for col in all_columns if (col not in target_columns)]
    y = dataset[target_columns].values
    #print(y.columns.tolist())
    X = dataset[remaining_columns].values
    eab_data = EABDataset(X, y, True)
    eab_trainloader = torch.utils.data.DataLoader(eab_data, batch_size=256, shuffle=True, num_workers=4)
    #print(len(X.columns.tolist()))

    imputer = NNImputer()
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(imputer.parameters(), lr=1e-4)

    for epoch in range(0, 100): # 5 epochs at maximum
    
        # Print epoch
        print(f'Starting epoch {epoch+1}')
    
        # Set current loss value
        current_loss = 0.0
        total_loss = 0.0
    
        # Iterate over the DataLoader for training data
        for i, data in enumerate(eab_trainloader, 0):
            
        # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 4))
            #print(inputs)
            #print(targets)

            # Zero the gradients
            optimizer.zero_grad()
      
            # Perform forward pass
            outputs = imputer(inputs)
            #print('outputs--------------------', outputs)
            #print('targets--------------------', targets)
      
            # Compute loss
            loss = loss_function(outputs, targets)
      
            # Perform backward pass
            loss.backward()
      
            # Perform optimization
            optimizer.step()
      
            # Print statistics
            current_loss += loss.item()
            total_loss += loss.item()
            #if i % 10 == 0:
                #print('Loss after mini-batch %5d: %.3f' %
                #     (i + 1, current_loss))
            #  current_loss = 0.0

        print(loss.item())
     # Process is complete.
    print('Training process has finished.')
