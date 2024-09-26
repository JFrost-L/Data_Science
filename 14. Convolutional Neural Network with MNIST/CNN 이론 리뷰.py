"""
오늘은 CNN에 대한 실습을 할 것

이론 review

output(H) = input(H) - Filter(H) + 1 : 이 때 stride과 padding는 1
output(W) = input(W) - Filter(W) + 1 : 이 때 stride과 padding는 1

3개의 채널이 있고 kernal이 3*3로 1개면 학습할 파라미터는 1*(3*3*3 + 1(bias))의 개수로 정해짐
feature map은 input에 kernal을 convolution한 결과로 feature map은 kernal의 개수와 같아진다.
이 feature map들의 합으로 output을 유도한다.
"""