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

class ImageClassificationModel_FC(nn.Module):
    def __init__(self):
        super(ImageClassificationModel_FC, self).__init__()
        print("FC 구조 초기화")
        self.fc1 = nn.Linear(3*300 * 300, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(0.2)
        self.fc6 = nn.Linear(64, 3)

    def forward(self, x):
        x = x.view(-1, 3*300 * 300)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout4(x)
        x = torch.relu(self.fc5(x))
        x = self.bn5(x)
        x = self.dropout5(x)
        x = self.fc6(x)
        return x

class ImageDataLoader:
    def __init__(self):
        dataset_path = './ProjectImages'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean, self.std = mean, std
        train_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.7, 0.7), shear=10),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        default_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        full_dataset = datasets.ImageFolder(root=dataset_path, transform=default_transforms)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        train_dataset.dataset.transform = train_transforms
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class ModelTrainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader, mean, std, epochs=20, learning_rate=0.00001):
        print("모델 트레이너 클래스 초기화")
        self.device = device  # 이미 device가 결정된 상태에서 모델을 GPU로 전송
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-7)
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
        print(f"New best accuracy: {accuracy:.4f}. Saving model...")
        model_save_path = f'FFNN_best_model_{accuracy:.4f}.pth'
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")
        self.best_model_path = model_save_path

    def load_best_model(self):
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
                    features, labels = features.to(self.device), labels.to(self.device)  # 데이터를 GPU로 전송
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
    
            print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
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
                features, labels = features.to(self.device), labels.to(self.device)  # 데이터를 GPU로 전송
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
        self.load_best_model()
        self.model.eval()
        test_loss = 0
        all_labels = []
        all_preds = []
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)  # 데이터를 GPU로 전송
                outputs = self.model(features.float())
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

    model = ImageClassificationModel_FC()
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
