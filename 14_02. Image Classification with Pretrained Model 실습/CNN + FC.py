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
import os

class ImageClassificationModel_CNN(nn.Module):
    def __init__(self):
        super(ImageClassificationModel_CNN, self).__init__()
        print("CNN 구조 초기화")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        # 여기서, 완전 연결 계층의 입력 크기를 적절하게 조정해야 합니다.
        self.fc1 = nn.Linear(1024 * 9 * 9, 512)
        self.bn11 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, 256)
        self.bn22 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 150
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 75
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 37
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 18
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # 9

        x = x.view(-1, 1024 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.bn11(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn22(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

class ImageDataLoader:
    def __init__(self):
        dataset_path = './ProjectImages'
        # 평균과 표준편차 계산
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean, self.std = mean, std
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 데이터 증강 적용
        train_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.3), scale=(1.2, 1.2), shear=10),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.train_loader.dataset.transform = train_transforms
        
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = -1

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class ModelTrainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader, mean, std, epochs=10, learning_rate=0.0001, smoothing=0.1):
        print("모델 트레이너 클래스 초기화")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = LabelSmoothingLoss(classes=3, smoothing=smoothing)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.mean = mean
        self.std = std
        self.train_losses = []
        self.val_losses = []
        self.best_accuracy = 0.0  # 최고 검증 정확도를 기록하는 변수
        self.best_model_path = None  # 최고 성능 모델 경로 저장
        
    def mixup_data(self, x, y, alpha=0.5):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def save_best_model(self, accuracy):
        """최고 성능 모델을 저장하는 함수"""
        print(f"New best accuracy: {accuracy:.4f}. Saving model...")
        model_save_path = f'CNN_best_model_{accuracy:.4f}.pth'
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")
        self.best_model_path = model_save_path  # 최고 성능 모델 경로 저장

    def load_best_model(self):
        """저장된 최고 성능 모델을 불러오는 함수"""
        if self.best_model_path:
            print(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        else:
            print("No model has been saved yet.")

    def train(self):
        print("모델 트레이너 클래스 train")
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch", leave=False) as pbar:
                for features, labels in self.train_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    mixed_features, labels_a, labels_b, lam = self.mixup_data(features, labels)
                    outputs = self.model(mixed_features)
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                    train_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
                    pbar.update(1)
    
            train_loss /= len(self.train_loader)
            val_loss, val_accuracy = self.validate()
    
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
    
            # Validation accuracy is now displayed as percentage
            print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
            # 최고 성능 모델 저장
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.save_best_model(val_accuracy)
    
        self.plot_loss()


    def validate(self):
        print("모델 트레이너 클래스 validation")
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        val_loss /= len(self.val_loader)
        val_accuracy = (correct / total) * 100  # Validation accuracy is now in percentage
        return val_loss, val_accuracy


    def plot_loss(self):
        print("그림 그리기")
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
        print("모델 트레이너 부분 test 모드")
        # 저장된 최고의 모델을 불러와서 평가
        self.load_best_model()
        self.model.eval()
        test_loss = 0
        all_labels = []
        all_preds = []
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features.float())
                outputs = outputs.view(labels.shape[0], 3, -1).mean(dim=2)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
    
        test_loss /= len(self.test_loader)
        accuracy = (correct / total) * 100  # Test accuracy is now in percentage
        print(f"Test Accuracy = {accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
    
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)


if __name__ == '__main__':
    begin_time = time.time()
    if torch.cuda.is_available():
        print("cuda")
    else:
        print("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ImageClassificationModel_CNN()
    print("모델 생성")
    model.to(device)
    print("모델 연결")
    dataloader = ImageDataLoader()
    print("ImageDataLoader!")
    
    trainer = ModelTrainer(model, device, dataloader.train_loader, dataloader.val_loader, dataloader.test_loader, dataloader.mean, dataloader.std)
    print("ModelTrainer!")
    
    trainer.train()
    print("train!")
    
    trainer.evaluate()
    print("evaluate!")
    
    end_time = time.time()
    
    print(f"elapsed_time={end_time - begin_time} seconds")
