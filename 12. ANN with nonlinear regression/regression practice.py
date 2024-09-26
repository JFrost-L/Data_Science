import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import matplotlib.pyplot as plt
import sys
from db_conn import *

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden1 = nn.Linear(1, 50)
        self.hidden2 = nn.Linear(50, 50)
        self.hidden3 = nn.Linear(50, 50)
        self.hidden4 = nn.Linear(50, 50)
        
        self.output = nn.Linear(50, 1)
        
    def forward(self, x):
        x = torch.relu(self.hidden1(x))  
        x = torch.relu(self.hidden2(x)) 
        x = torch.relu(self.hidden3(x))
        x = torch.relu(self.hidden4(x))     
       
        x = self.output(x) 
        return x
    
class DatasetGenerator:
    def __init__(self, n_points=1000, x_begin=0, x_end=10):
        self.n_points = n_points
        self.x_begin = x_begin
        self.x_end = x_end
        self.noise_level = 0.1

    # y = a*x + b
    def generate_linear(self, a=1, b=2):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a * x_values + b + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)
    
    # y = a*x^3 + b*x^2 + c*x + d
    def generate_polynomial(self, a=1, b=-3, c=-9, d=27):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a * (x_values ** 3) + b * (x_values ** 2) + c*x_values + d + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)
    
    # y = a*e^(b*x)
    def generate_exponential(self, a=1, b=2):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a * np.exp(b * x_values) + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

    # y = a + b*log(x)
    def generate_logarithmic(self, a=1, b=2):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a + b * np.log(x_values) + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

    # y = 1/(1+e^(a+b*x))
    def generate_sigmoid(self, a=1, b=2):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = 1 / (1 + np.exp(-(a + b * x_values))) + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

    # y = a*sin(b*x+c)
    def generate_sine(self, a=1, b=2, c=3):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a * np.sin(b * x_values + c) + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)


class ModelTrainer:
    def __init__(self, model, n_epochs=200, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        #gradient clipping : 기울기가 threshold를 넘지 않도록 클리핑!하는 필드 추가
        self.gradient_clip_val = 1000
        
        self.criterion = nn.MSELoss() #CrossEntropyLoss, BCELoss
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Adam, Adagrid, RMSprop
        
        self.losses = []

    def train(self, x_train, y_train):
        for epoch in range(self.n_epochs):
            self.model.train() #학습모드 지정
            self.optimizer.zero_grad() #에폭마다 GD값 초기화
            
            outputs = self.model(x_train) #현재 파라미터 세팅에 대한 결과 prediction
            loss = self.criterion(outputs, y_train) #손실값 계산
            loss.backward() #backward propagation, 파라미터별 gd 계산
            #gradient clipping 적용!
            #torch.nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step() #파라미터 값 업데이트
            
            self.losses.append(loss.item())

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {loss.item():.4f}')


class Visualizer:
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer

    def plot_loss(self):#loss 그래프 그리기
        plt.plot(self.model_trainer.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss During Training')
        plt.show()

    #예측 그래프 그리기
    def plot_prediction(self, x_train, y_train):
        with torch.no_grad():
            self.model_trainer.model.eval()  # 학습모드가 아닌 평가모드로 명시
            predicted = self.model_trainer.model(x_train).cpu().numpy()  # GPU 텐서를 CPU로 이동 후 NumPy로 변환
    
        # x_train을 CPU로 이동 후 정렬
        sorted_indices = np.argsort(x_train.cpu().numpy().flatten())  # x_train을 CPU로 이동 후 flatten
        sorted_x_train = x_train.cpu().numpy().flatten()[sorted_indices]  # x_train을 CPU로 이동 후 정렬
        sorted_predicted = predicted.flatten()[sorted_indices]
    
        # 시각화
        plt.figure()  # 새로운 그래프 창
        plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), label='Actual Data')  # 데이터를 CPU로 이동 후 시각화
        plt.plot(sorted_x_train, sorted_predicted, color='red', label='Fitted Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.show()

    
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')    
        
    model = RegressionModel().to(device)
    res = summary(model, input_size=(1, 1)) 
    print(res)
    
    generator = DatasetGenerator(n_points=10000, x_begin=-10, x_end=10)


    # x_train, y_train = generator.generate_linear()
    x_train, y_train = generator.generate_polynomial()
    # x_train, y_train = generator.generate_exponential()
    # x_train, y_train = generator.generate_logarithmic()#x의 범위 주의
    # x_train, y_train = generator.generate_sigmoid()
    # x_train, y_train = generator.generate_sine()
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    print(f"x_train={x_train}")
    print(f"y_train={y_train}")

    print(f"x_train.shape={x_train.shape}")
    print(f"y_train.shape={y_train.shape}")
    
    trainer = ModelTrainer(model, n_epochs=10000)
    trainer.train(x_train, y_train)


    visualizer = Visualizer(trainer)
    visualizer.plot_loss()
    visualizer.plot_prediction(x_train, y_train)    
