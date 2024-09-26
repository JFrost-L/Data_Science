"""
오늘은 pytorch로 분류할 것!

라이브러리 설명

분류 결과를 위해 confusion matrix 라이브러리 이용할 것
이 때 mysql db로 데이터 불러들일 것


분류 모델 클래스 생성
생성자와 forward를 override
생성자에서 디자인할 NN의 변수들을 초기화하고(이 때 layer도 정의해줌)
forward에서 초기화된 NN의 변수들을 이용해서 NN을 디자인(이 때 layer를 연결)

NN에서 분류할 때 중요한 것은 output node의 개수는 labeled의 개수와 동일해야함.

forward 결과로 output node를 각각 logit이라고 하는데 해당 logit의 값은 실수가 되는데(확률값이 아님)
결과적으로 각 output node들 중에서 가장 큰 값에 해당하는 class로 예측하도록 유도

이 때 torch.utils.data로부터 DataLoader, Dataset를 import해서
Dataset은 학습에 용이하게 전처리해주고
DataLoader로 그 dataset을 batch화 시켜서 공급

그래서 Dataset을 상속받아서 모델을 위한 Dataset 클래스를 정의
len()으로 데이터 개수를 리턴, getItem으로 index에 해당하는 feature와 labels을 튜플로 리턴

추가로 DataLoader도 클래스로 생성

db로부터 데이터를 읽어서 데이터를 loading
이 때 torch.tensor가 아닌 split_data()로 train, validate, test를 7:1.5:1.5로 비율 반영
이 때 validation data는 학습 도중에 evaluation을 하는 데이터

torch.tensor 대신에 Dataset 클래스로부터 데이터를 세팅
(이 때 레이블을 int가 아닌 long으로 한 이유는 데이터 유지를 위해서)
그리고 batch_size를 고려해서 DataLoader 클래스로부터 loader 객체들을 세팅
(이 때 train data만 shuffle=True 설정)

여기서 정의한 Dataset과 batch를 정의해서 DataLoader를 정의


ModelTrainer 클래스르 정의해서 학습을 위한 세팅
여기서 볼 것은 regression에서는 MSE를 loss_func로 설정했는데
classification에서는 CrossEntropy를 loss_func로 결정

train()에서 epoch의 수만큼 반복해서 train하는데 
with tqdm()을 이용하는데 
with으로 객체 사용 후 자동으로 제거
tqdm은 pytocrch는 tensorflow와 달리 
self.model(features,  float())은 logits으로 forward 결과를 반환
backward()로 오류역전파 진행
pbar는 tqdm을 통해서 set_postfix()에 결과를 출력하고 update()사용
tqdm은 터미널에서 진행 상황 바(progress bar)를 표시하는 라이브러리로
코드가 얼마나 진행되었는지 시각적으로 제시.
이는 특히 반복이 많이 일어나는 작업이나 데이터 처리 작업에서 유용
막대 그래프를 보기 싫다면 leave를 false로 변환


validate()로 학습 모드에서 평가모드로 전환 후에
경사하강을 하지 않고 테스트 진행
이 때 train_loss와 validate_loss를 비교해서 validate_loss가 더 적으면 더 반복해서 학습해도 됨

evaluate()로 실제 테스트 진행
gradient를 하지 않고 test_loader로부터 반복
loss를 계산해서 loss를 누적

logit이 각 output node이니까 계산해서 numpy list로 저장
확률값은 F.softmax()에 logit값을 주어 계산해서 numpy list로 저장

정답과 예측이 동일하면 tensor 형태로 되어 있으니 elementwise boolean 연산으로 정확도 연산

"""