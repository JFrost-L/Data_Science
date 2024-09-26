"""
라이브러리 설명
pytorch와 torchinfo를 추가로 설치하기
torchinfo은 모델을 요약해줌.

RegressionModel 클래스 정의
파이토치는 클래스 기반으로 구성되어 있어서 상속을 받아서 클래스를 만들 것(nn.Moduble)

생성자에서 부모 생성자도 마찬가지로 초기화
그리고 생성자 내에서 layer들을 정의하고 초기화하는 과정을 갖음
이 때 hidden layer와 output layer의 구조만 초기화 함.
이 때 nn.Linear(input, output)는 fully connected로 구성
nn.Linear(input 개수, output 개수)이니까 20개의 뉴런이 존재하는 layer
그 다음 hidden layer에는 nn.Linear(첫 hidden layer의 output 개수, output 개수)

forward(x) 메서드에서는 생성자에서 정의한 hidden layer들의 activation f를 정의하고
주어진 x인 input data에 대해서 forward propagation 구조를 설계 및 연결.
이 때 마지막 output layer의 activation f는 정의하지 않고 그대로 내보냄.

(시험문제)
해당 코드에 의하면 weight의 수는 20*20으로 400개이고
뉴런의 수는 20*2+1이며 bias의 수는 20*2로 파라미터의 수는 총 481이다.
즉, NN은 학습해야할 parameter가 너무 많이 필요해서 학습데이터가 많이 필요한 단점이 존재


데이터 만들어 내는 DatasetGenerator 클래스 정의 : 이 클래스로 훈련 데이터 셋을 생성
생성자에서는 노이즈를 포함한 초기값들을 초기화
init(n_points, x_begin, x_end)
n_point : 생성할 데이터 개수, [x_begin, x_end]로 데이터 값의 범위 정의

generate_linear(a, b) 메서드는 y=ax+b에 대한 메서드로 주어진 a와 b로 y=ax+b에서 a, b를 초기화

데이터 x_values를 초기에 정의한 값들의 제약을 고려해서 랜덤으로 한 개 생성
noise 값도 하나 랜덤으로 범위를 설정해서 생성
y_values = a*x_values + b + noise로 노이즈가 섞인 synthetic data인 y_values 유도
결과로 x_values(임의의 랜덤 입력)와 y_values(예측값)을 tensor 형태로 바꾸고 튜플로 리턴


학습하기 위한 ModelTrainer 클래스 정의
생성자에서는 생성한 model과
경사하강법을 위한 hyper parameter인 epochs의 수와 learning rate를 초기화
초기화할 때 중요한 것 2가지 설정이 존재
1. loss_function이 무엇인지 정의 여기서는 MSE
2. optimizer를 정의 여기서는 SGD
optimizer는 경사하강법 알고리즘 중에서 무엇을 쓸 지 결정하는 것
여기서 추가로 epochs마다 loss값을 저장할 리스트를 초기화

train 메서드에서는 x, y 데이터를 제공받아 학습을 진행
epoch를 반복해서 돌아야하는데 그 때마다
큰 틀인 구조는 내가 설계하지만 실제 내부는 파이토치가 진행
(시험문제) 메서드들의 대한 이해
self.model.train() : 학습 모드를 명시
self.optimizer.zero_grad() : epoch을 한 번 진행할 때마다 GD값을 초기화
outputs = self.model(x_Train) : 현재 주어진 x에 대해서 예측값을 계산하고 그 결과를 리턴
loss = self.criterion(outputs, y_train)으로 loss값을 계산
loss.backward()로 오류 역전파로 파라미터별 gd 계산
self.optimizer.step() : 파라미터값을 업데이트로 학습
그리고 loss값 저장하고 epoch을 10번 할 때마다 출력하도록 설정


마지막 코드는 학습 결과에 대한 plot을 그리기
Visualizer라는 클래스로 진행
첫 번째 그래프는 epoch별 training loss값을 나타내는 그래프를 제시
두 번째 그래프는 sorting한 모든 x_values에 대한 y_values set을 그래프에 찍고
실제 그래프는 빨간색 직선

중요한 부분은 해당 코드에서 nonlinear한 모양의 그래프를 형성하는
데이터에 대해서는 작동하지 않는다.
"""