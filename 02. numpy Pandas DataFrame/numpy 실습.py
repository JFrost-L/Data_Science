# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:35:24 2023

@author: L
"""

'''
1. numpy
numpy는 Homogeneous, multidimensional array
즉 같은 타입의 다차원 행렬에 적절
ndim : numpy array 정의할 때 차원을 알 수 있음
shape : 차원의 행*열을 알 수 있음
size : 차원 상관없이 element 개수 알 수 있음
dtype : 해당 넘파이의 데이터 타입을 알 수 있음
itemsize : element(item)의 타입 사이즈(byte)

import numpy as np
a = np.arange(15).reshape(3,5)
#arange(n)은 0~14까지의 숫자를 생성한 후 numpy array를 만들어 리턴
#reshape(n, m)으로 n행 m열로 행렬 변환
print(a)
print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.itemsize)
print(a.size)
print(type(a))
'''

'''
2. numpy 초기화
넘파이는 파이썬의 리스트와 다르게 처음에 데이터를 잡고(초기화) 작동시켜야 함

리스트와 넘파이는 서로 호환이 안되어 캐스팅을 해야함
a = np.array(10, 100, 10)
->리스트화 : a= list(a)
->넘파이화 : a= np.array(a)

import numpy as np
a = np.array([[1,2,3], [4,5,6]])
print(a)
#2 * 3의 numpy array 생성

b = np.zeros((3,4))
print(b)
#element가 0인 3*4 numpy array 생성

c = np.ones( (2,3,4), dtype = np.int64 )
print(c)
#element가 1인 2*3*4 tensor를 생성하는데 타입을 np.int64로 정의

d = np.empty( (2,3) )
#랜덤값의 2*3의 행렬을 생성(여기서는 0.으로 설정)
print(d)

e = np.arange(10,100, 10)
#10~100까지(100 포함 x) step이 10인 1차원 numpy array 생성
print(e)
'''

'''
3. numpy array의 연산
element간의 연산 존재 : “elementwise” operations
-> 딥러닝에서 병렬화해서 연산할 때 많이 사용
elementwise operation은 벡터화 연산 적용 가능: for loop 보다 빠른 실행

import numpy as np
A = np.array( [20,30,40,50] )
print(A)
B = np.arange(4)
print(B)
C = A-B
print(C)
#shape가 동일해야지 성립. 짝이 맞는 element간의 연산 수행
D = B**2
print(D)
#element 단위로 제곱 연산
print(A<35) 
#A의 각 element 단위로 비교 연산해서 numpy 반환
'''

'''
4. list vs ndarray

list
• 여러가지 타입의 원소
• linked List 구현
• 메모리 용량이 크고 속도가 느림
• 벡터화 연산 불가

ndarray
• 동일 타입의 원소
• contiguous memory layout
• 메모리 최적화(메모리가 continuous 하게 잡히기에), 계산 속도 향상
• 벡터화 연산 가능

ndarray 벡터화 계산(vectorized computation) 효과 Check

#1. 일반 리스트 연산
import time
a = range(10000000)
b = range(10000000)
c = []
start = time.time()
for i in range(10000000):
    c.append(a[i]*b[i])
end = time.time()
print("elasped time =", end-start)

#2. 넘파이의 벡터화 연산
import numpy as np
import time
a = np.arange(10000000)
b = np.arange(10000000)
start = time.time()
c = a*b
end = time.time()
print("elasped time =", end-start)
'''

'''
Lab01
import numpy as np
import random as r
import time as t

start = t.time()
numList = [r.randint(1, 1000+1) for _ in range(1000000)]
modifiedNumList = [1 if n>500 else 0 for n in numList ]
end = t.time()
print("elasped time =", end-start)


start = t.time()
numArr = np.random.randint(0, 1000+1, size=1000000)
modifiedNum_Arr = np.where(numArr>500, 1, 0)
end = t.time()
print("elasped time =", end-start)
'''


'''
5. numpy 행렬 단위 연산
ndarray matrix-wise operations : 행렬 곱 및 벡터의 내적

'''
import numpy as np
A = np.array ( [[1,1],[0,1]] )
B = np.array( [[2,0],[3,4]] )
C = A*B
# elementwise product : 행렬의 원소간의 곱
D = A@B
# matrix product : 행렬의 곱

print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")
print(f"D = {D}")

X = [1,2,3,4]
Y = [1,0,1,0]
F = np.inner(X,Y)
# inner product : 벡터의 내적

print(f"X = {X}")
print(f"Y = {Y}")
print(f"F = {F}")