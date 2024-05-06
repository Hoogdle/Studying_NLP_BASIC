### 텐서 조작하기 ###

# NLP에서의 3차원 텐서는 [배치사이즈,문장길이,단어벡터차원]으로 이루어져 있다.
# *배치사이즈는 '전체 데이터' 개별 데이터 하나씩 가져오는 것이 아닌 '배치 크기'만큼 한번에 가져와 학습한다.


### 넘파이 텐서 실습

import numpy as np

### 1차원 텐서
t = np.array([0,1,2,3,4,5,6]) #list를 np.array로 1차원 array로 변환
print(t) #[0 1 2 3 4 5 6]

# t의 '차원'과 '크기' 출력
# .ndim은 차원을 출력하며 1차원 == 벡터, 2차원 == 행렬, 3차원 == 텐서
print('Rank of t: ',t.ndim) #Rank of t:  1
print('Shape of t : ',t.shape) #Shape of t :  (7,)

# 원소별 접근, 슬라이싱도 가능!
print('t[0] t[1] t[-1] = ',t[0],t[1],t[-1]) #t[0] t[1] t[-1] =  0 1 6
print('t[2:5] t[4:-1] = ', t[2:5],t[4:-1]) #t[2:5] t[4:-1] =  [2 3 4] [4 5]


### 2차원 텐서
t = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(t)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
print('Rank of t : ',t.ndim)
print('Shape of t : ',t.shape)
# Rank of t :  2
# Shape of t :  (4, 3)

### Pytorch에서의 텐서
import torch

### 1차원 텐서
t = torch.FloatTensor([0,1,2,3,4,5,6])
print(t) 
#tensor([0., 1., 2., 3., 4., 5., 6.])

print(t.dim()) #1
print(t.shape) #torch.Size([7]) #shape
print(t.size()) #torch.Size([7]) #shape(위와 동일)

print(t[0],t[1],t[-1]) #tensor(0.) tensor(1.) tensor(6.)
print(t[2:5],t[4:-1]) #tensor([2., 3., 4.]) tensor([4., 5.])
print(t[:2],t[3:]) #tensor([0., 1.]) tensor([3., 4., 5., 6.])


### 2차원 텐서
t = torch.FloatTensor([[1,2,3],
                       [4,5,6],
                       [7,8,9],
                       [10,11,12]])
print(t)
# tensor([[ 1.,  2.,  3.],
#         [ 4.,  5.,  6.],
#         [ 7.,  8.,  9.],
#         [10., 11., 12.]])

print(t.dim()) #2
print(t.size()) #torch.Size([4, 3])

print(t[:,1]) #tensor([ 2.,  5.,  8., 11.])
print(t[:,1].size()) #torch.Size([4]) (위의 경우의 크기)

print(t[:,:-1])
# tensor([[ 1.,  2.],
#         [ 4.,  5.],
#         [ 7.,  8.],
#         [10., 11.]])


### 브로드 캐스팅
# 행렬의 덧셈,뺄셈 에서는 두 행렬 A,B의 크기가 같아야 하며 곱셈을 할 때는 A의 열과 B의 행의 크기가 같아야 한다.(in 수학적개념)
# 딥러닝에서는 크기가 다른 텐서에 대한 사칙연산을 하는 경우가 불가피.... => 자동으로 크기를 맞춰주는 [브로드 캐스팅]

m1 = torch.FloatTensor([[3,3]])
m2 = torch.FloatTensor([[2,2]])
print(m1+m2) #tensor([[5., 5.]])

### 크기가 다른 텐서간 연산
# Vector + Scalar
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([3]) # [3] -> [3,3]
print(m1+m2) #tensor([[4., 5.]])

print(m1.size()) #torch.Size([1, 2])
print(m2.size()) #torch.Size([1]) #연산을 위해 m2를 [1,2]로 브로드 캐스팅!

# Vector [2,1] x Vector [1,2]

m1 = torch.FloatTensor([[1],[2]])
m2 = torch.FloatTensor([[1,2]])
print(m1.size()) #torch.Size([2, 1])
print(m2.size()) #torch.Size([1, 2])
print(m1+m2)
# tensor([[2., 3.],
#         [3., 4.]])

# m1
# [1]
# [2]
# ==> [[1,1]
#      [2,2]]

# m2
# [1,2]
# ==> [[1,2]
#      [1,2]]

# 브로드캐스팅은 '자동'으로 수행되므로 사용에 주의하도록 하자!


### 행렬곱 vs 곱셈의 차이
# 행렬곱셈(.matmul) 원소별 곱셈(.mul)

m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print('m1 size : ',m1.shape) #m1 size :  torch.Size([2, 2])
print('m2 size : ',m2.shape) #m2 size :  torch.Size([2, 1])
print(m1.matmul(m2)) # matmul은 '행렬곱'을 수행한다.
# tensor([[ 5.],
#         [11.]])

### * or .mul() => 원소별 곱으로 동일한 크기의 행렬이 동일한 위치에 있는 원소별곱
# 행렬의 크기가 동일하지 않으면 브로딩캐스팅 후 연산된다.
print(m1*m2)
# tensor([[1., 2.],
#         [6., 8.]])
print(m1.mul(m2))
# tensor([[1., 2.],
#         [6., 8.]])



### 평균(Mean)
t = torch.FloatTensor([1,2])
print(t.mean()) #tensor(1.5000)

t = torch.FloatTensor([[1,2],[3,4]])
print(t.mean()) #tensor(2.5000)

print(t.mean(dim=0)) # 첫 번째 차원(dim=0)을 제거 => (2,2) -> (1,2)==(2,)==벡터
# tensor([2., 3.])
print(t.mean(dim=1))
# tensor([1.5000, 3.5000])
# 차원이 삭제되면서 삭제되는 데이터가 생존하는 데이터에 정사영된다. 그 후 평균이 계산됨.
print(t.mean(dim=-1)) # 마지막 차원. 즉, 열의 차원을 제거
# tensor([1.5000, 3.5000])



### 덧셈(Sum)
t = torch.FloatTensor([[1,2],[3,4]])
print(t)
# tensor([[1., 2.],
#         [3., 4.]])
print(t.sum()) # 단순 모든 원소의 합
# tensor(10.)
print(t.sum(dim=0)) # 행을 제거하여 덧셈
# tensor([4., 6.])
print(t.sum(dim=1)) # 열을 제거하여 덧셈
# tensor([3., 7.])
print(t.sum(dim=-1)) # 마지막 차원(열)을 제거하여 덧셈
# tensor([3., 7.])


### 최대(MAX)와 아그맥스(ArgMAX)
# MAX는 원소의 최대값을, ArgMAX는 최댓값의 원소의 인덱스를 리턴
t = torch.FloatTensor([[1,2],[3,4],[5,6]])
print(t)
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.]])
print(t.max()) #tensor(6.)

# .max() 함수의 인자로 'dim'을 주면 argmax도 함께 반환된다.
print(t.max(dim=0)) # 없애는 차원(행)을 기준으로 봤을 때 [5,6]은 각 열의 2번째 행에 존재
# torch.return_types.max(
# values=tensor([5., 6.]),
# indices=tensor([2, 2]))

print(t.max(dim=1)) # 없애는 차원(열)을 기준으로 봤을 때 [2,4,6]은 각 행의 1번째 열에 존재
# torch.return_types.max(
# values=tensor([2., 4., 6.]),
# indices=tensor([1, 1, 1]))

print('Max : ',t.max(dim=0)[0]) #Max :  tensor([5., 6.])
print('Arg Max : ',t.max(dim=0)[1]) #Arg Max :  tensor([2, 2])