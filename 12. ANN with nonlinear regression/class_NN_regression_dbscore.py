import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import sys

from db_conn import *

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden1 = nn.Linear(2, 20)
        self.hidden2 = nn.Linear(20, 20)
        self.hidden3 = nn.Linear(20, 20)

        self.output = nn.Linear(20, 1)
        
    def forward(self, x):
        x = torch.relu(self.hidden1(x))  
        x = torch.relu(self.hidden2(x))  
        x = torch.relu(self.hidden3(x)) 
        
        y = self.output(x) 
        
        #print(f"x={x}, y={y}")
        #sys.exit()
        
        return y


class DatasetGenerator:
    def __init__(self):
        self.conn, self.cur = open_db()

    def load_dbscore_data(self):
        sql = "select * from db_score;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
    
        self.X = [ (t['midterm'], t['final']) for t in data ]
        self.X = np.array(self.X)

    
        self.y = [ t['score'] for t in data]
        self.y = np.array(self.y)  
        self.y = self.y.reshape(-1,1)
        
        #print(f"self.X={self.X}")
        #print(f"self.y={self.y}")
        #sys.exit()

        return torch.tensor(self.X, dtype=torch.float32), torch.tensor(self.y, dtype=torch.float32)        


class ModelTrainer:
    def __init__(self, model, n_epochs=200, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []

    def train(self, x_train, y_train):
        for epoch in range(self.n_epochs):
            self.model.train() #학습모드 지정
            self.optimizer.zero_grad() #에폭마다 GD값 초기화
            outputs = self.model(x_train)

            loss = self.criterion(outputs, y_train)
            loss.backward()
            
            self.optimizer.step()
            self.losses.append(loss.item())

     
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {loss.item():.4f}')
        

class Visualizer:
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer

    def plot_loss(self):
        plt.plot(self.model_trainer.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss During Training')
        plt.show()

    def plot_prediction(self, x_train, actual_y):
        with torch.no_grad():
            self.model_trainer.model.eval()
            predicted_y = self.model_trainer.model(x_train).numpy()

        x_axis = np.arange(len(actual_y))
        plt.scatter(x_axis, actual_y, color='red', label='Actual Y')
        plt.scatter(x_axis, predicted_y, color='blue', label='Predicted Y')
        plt.xlabel('Data Points')
        plt.ylabel('Y values')
        plt.title('Actual vs Predicted Y values')
        plt.legend()
        plt.show()


        '''
        # y_train으로 정렬
        sorted_indices = np.argsort(y_train.numpy().flatten())
        sorted_x_train = x_train.numpy().flatten()[sorted_indices]
        sorted_predicted = predicted.flatten()[sorted_indices]
        '''



if __name__ == '__main__':
    generator = DatasetGenerator()

    x_train, y_train = generator.load_dbscore_data()

    
    print(f"x_train.shape={x_train.shape}")
    print(f"y_train.shape={y_train.shape}")
    #sys.exit()    
    
    model = RegressionModel()
    trainer = ModelTrainer(model, n_epochs=1000)
    trainer.train(x_train, y_train)
    
    visualizer = Visualizer(trainer)
    visualizer.plot_loss()
    visualizer.plot_prediction(x_train, y_train)
