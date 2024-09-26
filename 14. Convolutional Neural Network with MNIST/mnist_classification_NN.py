import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import time

import matplotlib.pyplot as plt

#MNIST는 손으로 숫자를 쓴 28*28 이미지 데이터를 0~9사이의 숫자로 classification하기 위한 데이터셋
class MNISTClassificationModel_FC_only(nn.Module):
    #Fully connected로만 수행하는 모델
    def __init__(self):
        super(MNISTClassificationModel_FC_only, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        #input은 데이터가 28*28의 이미지
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        #마지막 output은 숫자가 0~9의 종류가 있기에 10

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        #이를 마지막에 fully connnected를 하기 위해서 1차원으로 변환
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        #이렇게 나온 결과 x는 logit!
        
        return x


class MNISTClassificationModel_CNN(nn.Module):
    #CNN으로 수행하는 모델
    def __init__(self):
        super(MNISTClassificationModel_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        #1=채널수, 32=출력채널(필터의 수), 필터크기 -> (32, H, W) 
        #즉, 필터의 개수가 32개라서 결과로 나오는 feature map의 개수가 1개의 input 이미지로부터 32개가 도출
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # output: (64, H, W) 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # output: (H/2, W/2)

        # self.fc1 = nn.Linear(32 * 14 * 14, 64) # 첫 번째 feature map을 위한 연산을 디자인       
        self.fc1 = nn.Linear(64 * 7 * 7, 64) 

        self.fc2 = nn.Linear(64, 10)  # 두 번째 feature map을 위해 연산을 디자인 이는 곧 output을 위한 디자인
        # 출력 계층으로 바로 연결

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # (32, 14, 14)
        #1채널을 32채널로 바꾸고 pooling연산으로 14*14로 감소하는 연산
        x = self.pool(torch.relu(self.conv2(x))) # (64, 7, 7)
        #32채널을 64채널로 바꾸고 pooling 연산으로 7*7로 감소하는 연산

        # x = x.view(-1, 32 * 14 * 14) # Flatten        
        x = x.view(-1, 64 * 7 * 7) # Flatten
        #이를 마지막에 fully connnected를 하기 위해서 1차원으로 변환
        
        x = torch.relu(self.fc1(x))
        #최종 결과 내기
        x = self.fc2(x)

        return x



class MNISTDataLoader:
    #batch화 하기 위한 data loader
    def __init__(self):        

        #이미지를 처리하기 위해서는 transform 정규화를 필수적으로 해야 성능이 좋음.
        transform = transforms.Compose([
            transforms.ToTensor(), #픽셀값을 [0,1] 사이로 scaling
            transforms.Normalize((0.5,), (0.5,)) #픽셀값을 mean=0.5, std=0.5로 정규분포로 정규화
        ])
        #MNIST data를 파이토치로 다운로드

        full_train_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
        #여기서 transform은 일종의 전처리하는 parameter를 이용해 full train dataset 설정
        train_size = int(0.7 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size 
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
        test_dataset = datasets.MNIST(root='./', train=False, download=True, transform=transform)
        #data set을 train, validation, test로 분기
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        #random으로 batch처리를 하기 위해서 batch_size만큼 데이터를 분리해서 loader 설정

class ModelTrainer:
    #모델 트레이너 클래스 설정
    def __init__(self, model, train_loader, val_loader, test_loader, epochs=10, learning_rate=0.001):
        # GPU가 사용 가능한지 확인하고 device 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #학습하기 위한 parameter 초기화
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        #loss func은 multiclass classification이기에 CrossEntropy
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        #optimizer는 Adam!
        self.epochs = epochs

        self.train_losses = []
        self.val_losses = []

    def train(self):
        for epoch in range(self.epochs):
            #주어진 epochs만큼 반복해서 학습
            self.model.train()
            train_loss = 0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch", leave=False) as pbar:
                for features, labels in self.train_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)  # __call__()
                    loss = self.criterion(outputs, labels)
                    train_loss += loss.item()
                    #train_loss를 누적
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
                    pbar.update(1)

            train_loss /= len(self.train_loader)
            #누적한 train_loss를 평균
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)            
            
            print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # self.plot_loss()

    def validate(self):
        #validation 과정으로 평가모드로 설정
        self.model.eval()
        val_loss = 0
        with torch.no_grad():#평가모드일 때는 gradient 사용 x
            for features, labels in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device) 
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                #validation loss를 누적
        
        val_loss /= len(self.val_loader)
        #누적한 validatioin loss를 평균
        return val_loss


    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.show()

    
    def evaluate(self):
        #진짜 test를 위해 평가모드 설정
        self.model.eval()
        test_loss = 0
        all_labels = []
        all_preds = []
        output_list = []
        output_prob_list = []  # 클래스별 확률을 저장할 리스트

        total = 0
        correct = 0
        

        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                #print(f"features={features}")
                #print(f"labels={labels}")                
                outputs = self.model(features.float())
                #forward를 통해서 예측값 구하기
                #print(f"outputs={outputs}")
              
                
                loss = self.criterion(outputs, labels)
                #예측과 정답의 loss function으로 차이 구하기
                test_loss += loss.item()
                
                output_list.extend(outputs.cpu().numpy())
                
                output_prob = F.softmax(outputs, dim=1)
                #logit을 확률분포로 변환
                output_prob_list.extend(output_prob.cpu().numpy())

                _, predicted = torch.max(outputs, 1)
                #logit들 중에 가장 큰 것으로 분류할 것이기에 예측 뽑아내기
                total += labels.size(0)
                #정확률을 위한 전체 개수
                correct += (predicted == labels).sum().item()
                #정답의 개수 구하기
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                #all_labels.extend(labels.cpu().numpy())
                #all_preds.extend(predicted.cpu().numpy())

                # for f, l, p, o, prob in zip(features, labels, predicted, outputs, output_prob):
                #     # if l == p:
                #     #     continue
                #     #이는 labeled와 prediction이 같은 경우는 continue하도록 설정해서 틀린 경우만 출력
                #     plt.imshow(f.squeeze(), cmap='gray')
                #     plt.title(f'Predicted: {p}, Actual: {l}')
                #     plt.show()            
                #     # print(f"{o}->{prob}: {l} -> {p}")


        test_loss /= len(self.test_loader)
        

        print(f"Accuracy = {correct/total: .4f}")

        cm = confusion_matrix(all_labels, all_preds)
        print(f"Test Loss: {test_loss:.4f}")
        print("Confusion Matrix:")
        print(cm)
        

if __name__ == '__main__':
    begin_time = time.time()
    
    # model = MNISTClassificationModel_FC_only()
    model = MNISTClassificationModel_CNN()
    
    dataloader = MNISTDataLoader()
    
    trainer = ModelTrainer(model, dataloader.train_loader, dataloader.val_loader, dataloader.test_loader, epochs=5)
    trainer.train()
    trainer.evaluate()

    end_time = time.time()
    
    print(f"elapsed_time={end_time - begin_time} seconds")
    
    
    
    