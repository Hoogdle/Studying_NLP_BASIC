### 선형 회귀 구현

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) # 코드를 재실행 하더라도 같은 결과가 나오도록 랜덤 시드 부여

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

print(x_train)
print(x_train.shape)

print(y_train)
print(y_train.shape)

# tensor([[1.],
#         [2.],
#         [3.]])
# torch.Size([3, 1])
# tensor([[2.],
#         [4.],
#         [6.]])
# torch.Size([3, 1])

W = torch.zeros(1,requires_grad=True) # 가중치를 0으로 초기화 + 학습을 통해 변경되는 변수임을 명시(requires_grad = True)
print(W) #tensor([0.], requires_grad=True)
b = torch.zeros(1,requires_grad=True)
print(b) #tensor([0.], requires_grad=True)

# 현재의 가중치 세팅, 선형식은 다음과 같다.
# y = 0*x + 0

# 학습에서 사용할 가설
hypothesis = x_train*W + b
print(hypothesis)
# tensor([[0.],
#         [0.],
#         [0.]], grad_fn=<AddBackward0>)


# 비용 함수 선언
cost = torch.mean((hypothesis-y_train)**2)
print(cost) #tensor(18.6667, grad_fn=<MeanBackward0>)


# 경사 하강법 설정(옵티마이저 설정)
optimizer = optim.SGD([W,b],lr=0.01) # 학습 대상인 W와 b가 SGD의 입력으로 들어감

# optimizer.zero_grad로 미분을 통해 얻은 기울기를 0으로 초기화
# cost.backward() : W와 b에 대한 기울기가 계산됨
# optimizer.step() : optimizer의 인수로 들어간 W,b의 각각의 기울기에 lr를 곱하여 가중치 업데이트



### 전체 코드 ### 

x_train = torch.FloatTensor([[1],[2],[3]])
t_train = torch.FloatTensor([[2],[4],[6]])

W = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)
