"""
라이브러리 설명
pytorch와 torchinfo를 추가로 설치하기
torchinfo은 모델을 요약해줌.


nonlinear의 경우에는 learning rate의 조정과
일부 하이퍼 파라미터의 값을(epoch 수, hidden layer의 수와 뉴런 수) 조정해서 학습의 성능을 올릴 수 있다.


여기서 gradient clippping이라는 개념을 적용했는데 이는 gradient가 너무 커서
일정 threshold가 넘으면 해당 threshold로 gradient를 fix하는 기법이다.
이는 주로 RNN계열에서 gradient vanishing이나 gradient exploding이 많이 발생하는데,
gradient exploding을 방지하여 학습의 안정화를 도모하기 위해 사용하는 방법
clipping은 gradient의 L2norm(norm이지만 보통 L2 norm사용)으로 나눠주는 방식
clipping이 없으면 gradient가 너무 뛰어서 global minimum에 도달하지 못하고
엉뚱한 방향으로 향하게 되지만, clipping을 하게 되면 gradient vector가 방향은 유지하되
적은 값만큼 이동하여 도달하려고 하는 곳으로 안정적으로 내려가게 된다. 
특히 optimizer 설정을 SGD로 한 경우 많이 사용 Adam은 많이 사용하지는 않음.

nonlinear 함수 중에서 exponential의 경우는 loss가 커도 그래프를 그려서 보게 되면
학습이 잘 되었다는 것을 알 수 있는데 exponential이 운용하는 값들의 범주를 생각해보면
loss가 커도 exponential 개념에서는 그렇게 크지 않을 수도 있다는 것을 시사한다.


epoch의 수를 늘릴 필요가 있는지 없는지의 여부는 loss 그래프를 보고
그 그래프의 후반부가 안정적이지 않고 일정한 방향으로
즉, 감소하는 방향으로 그래프가 내려가는 경우 epoch를 늘려서 안정적인 상황까지 그래프를 유도한다.


로그함수의 경우는 x의 범위가 음수이면 안된다.
시그모이드 함수의 경우에는 noise를 0.1로 줄이자. 

sine 함수의 경우는 layer의 수와 뉴런의 수를 충분히 많이 주고 데이터를 많이 주면 가능
"""