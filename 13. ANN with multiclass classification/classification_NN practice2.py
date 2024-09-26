import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from tqdm import tqdm

from db_conn import *
import sys

class IrisClassificationModel(nn.Module):
    def __init__(self):
        
        self.number_of_input_features = 3
        self.number_of_classes = 3
        
        super(IrisClassificationModel, self).__init__()
        
        self.hidden1 = nn.Linear(self.number_of_input_features, 20)
        self.hidden2 = nn.Linear(20, 20)
        
        self.output = nn.Linear(20, self.number_of_classes)    
    
    def forward(self, x):
        x = torch.relu(self.hidden1(x))  
        x = torch.relu(self.hidden2(x))  
       
        x = self.output(x) 
        return x 


class IrisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    
    
class IrisDataLoader: 
    def __init__(self):        
        self.conn, self.cur = open_db()

    def load_iris_data_from_db(self):
        sql = "select * from iris;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        #print("data=", data)
    
        self.X = [ (t['sepal_length'], t['sepal_width'], t['petal_length'] ) for t in data ]
        #self.X = [ (t['sepal_length'], t['sepal_width'] ) for t in data ]
        #self.X = [ (t['sepal_length'], t['petal_length'] ) for t in data ]

        self.X = np.array(self.X)
    
        self.y =  [0 if t['species'] == 'setosa' else 1 if t['species'] == 'versicolor' else 2 for t in data]
        #self.y =  [ 1 if t['species'] == 'setosa' else 0 for t in data]
        self.y = np.array(self.y)    

        #return torch.tensor(self.X, dtype=torch.float32), torch.tensor(self.y, dtype=torch.int)      

    def split_data(self):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        train_dataset = IrisDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
        val_dataset = IrisDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
        test_dataset = IrisDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
        
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

    

class ModelTrainer:
    def __init__(self, model, dataloader, learning_rate=0.001, epochs=50):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_loader = dataloader.train_loader
        self.val_loader = dataloader.val_loader
        self.test_loader = dataloader.test_loader

        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch", leave=False) as pbar:
                for features, labels in self.train_loader:
                    outputs = self.model(features.float())  # __call__()
                    loss = self.criterion(outputs, labels)
                    train_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
                    pbar.update(1)

            train_loss /= len(self.train_loader)
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in self.val_loader:
                outputs = self.model(features.float())
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(self.val_loader)
        
        return val_loss


    def evaluate(self):
        self.model.eval()
        test_loss = 0
        all_labels = []
        all_preds = []
        output_list = []
        output_prob_list = []  # 클래스별 확률을 저장할 리스트
        label_name = ["setosa", "versicolor", "virginica"]
        total = 0
        correct = 0
        

        with torch.no_grad():
            for features, labels in self.test_loader:
                #print(f"features={features}")
                #print(f"labels={labels}")                
                outputs = self.model(features.float())
                #print(f"outputs={outputs}")
              
                
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                output_list.extend(outputs.numpy())
                
                output_prob = F.softmax(outputs, dim=1)
                output_prob_list.extend(output_prob.numpy())

                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item() # 불리언 텐서, True -> 1, False -> 0

                all_labels.extend(labels.numpy())
                all_preds.extend(predicted.numpy())
                #all_labels.extend(labels.cpu().numpy())
                #all_preds.extend(predicted.cpu().numpy())


        test_loss /= len(self.test_loader)
        
        for l, p, o, prob in zip(all_labels, all_preds, output_list, output_prob_list):
            print(f"{o}->{prob}: {label_name[l]} -> {label_name[p]}")

        print(f"Accuracy = {correct/total: .4f}")

        # 혼동 행렬 계산
        cm = confusion_matrix(all_labels, all_preds) # row = label, column = predicted
        print(f"Test Loss: {test_loss:.4f}")
        print("Confusion Matrix:")
        print(cm)



if __name__ == '__main__':
    model = IrisClassificationModel()

    dataloader = IrisDataLoader()
    dataloader.load_iris_data_from_db()
    dataloader.split_data()

    trainer = ModelTrainer(model, dataloader)
    trainer.train()
    trainer.evaluate()
    

















