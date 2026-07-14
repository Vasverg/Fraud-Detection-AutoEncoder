import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Data Prep
df = pd.read_csv('creditcard.csv')
df = df.drop(columns = ['Time'])
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
train_filtered = df[df['Class'] == 0].sample(frac = 0.8).copy() #80% training
test_df = df.drop(train_filtered.index) #20% test

#Features and Labels
train_feats = train_filtered.drop(['Class'], axis = 1).values
test_feats = test_df.drop(['Class'], axis = 1).values
labels = test_df['Class'].values

train_tensors = torch.from_numpy(train_feats).float().to(device)
test_tensors = torch.from_numpy(test_feats).float().to(device)

#DataLoader + Input Size for the AE
loader = DataLoader(TensorDataset(train_tensors), batch_size = 256, shuffle = True)
input_size = train_feats.shape[1]

#Autoencoder
class FraudDetectorAE(nn.Module):
   def __init__(self, input_size):
      super().__init__()
      self.encoder = nn.Sequential(
         nn.Linear(input_size, 16), nn.ReLU(),
         nn.Linear(16, 8), nn.ReLU(),
         nn.Linear(8, 4), nn.ReLU()
     )
      self.decoder = nn.Sequential(
         nn.Linear(4, 8), nn.ReLU(),
         nn.Linear(8, 16), nn.ReLU(),
         nn.Linear(16, input_size)
      )

   def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)

      return decoded
   
model = FraudDetectorAE(input_size).to(device)
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#Training
epochs = 15
model.train()

for epoch in range(epochs):
   train_loss = 0.0
   for batch in loader:
      inputs = batch[0]
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, inputs)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * inputs.size(0)
      
   epoch_loss = train_loss / len(loader.dataset)
   print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.5f}')

#Testing
model.eval()

with torch.no_grad():
   recons = model(test_tensors)
   recons_error = torch.mean(torch.pow(test_tensors - recons, 2), dim = 1).cpu().numpy()

#Threshold
threshold = np.percentile(recons_error, 98.8)
pred = (recons_error > threshold).astype(int)

#Plots + Metrics
fraud = np.where(labels == 1 )[0]
plt.figure(figsize = (16, 8))
plt.plot(recons_error, label = "Reconst. Error", alpha = 0.25, color = 'blue')
plt.scatter(fraud, recons_error[fraud], color = 'red', label = 'Actual Frauds', s = 4)
plt.axhline(threshold, color = 'black', linestyle = '--', label = 'Threshold')
plt.title('Fraud Detection')
plt.xlabel('Transaction')
plt.ylabel('Reconst. Error')
plt.legend()
plt.show()

precision = metrics.precision_score(labels, pred, pos_label = 1)
recall = metrics.recall_score(labels, pred, pos_label = 1)
print(f"---Metrics Class: Frauds---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
