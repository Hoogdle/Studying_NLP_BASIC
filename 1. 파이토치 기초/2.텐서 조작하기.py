import torch
import numpy as np

### .view() : 원소의 수는 유지, 텐선의 크기는 변경

# cf) Numpy의 array를 Tensor로 변환 가능!
t = np.array([[[0,1,2],
              [3,4,5]],
              [[6,7,8],
              [9,10,11]]])
ft = torch.FloatTensor(t)   

print(ft.shape) #torch.Size([2, 2, 3])

# [-1,3]에서 -1은  첫 번째 차원의 설정을 파이토치에게 맡긴다는 의미, 3은 두 번째 차원의 크기는 3으로 설정한다는 의미
# => 차원을 [?,3]의 크기로 변경하라는 의미
print(ft.view([-1,3])) # ft라는 텐서를 (?,3) 텐서로 변경
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.],
#         [ 6.,  7.,  8.],
#         [ 9., 10., 11.]])
print((ft.view([-1,3])).shape) #torch.Size([4, 3])

# <.view() 함수의 규칙>
# 1. 변경 전과 변경 후의 '원소의 갯수'는 유지
# 2. view의 사이즈가 -1로 설정되어 있다면 다른 차원으로 부터 해당 값을 유추한다.

### .view()로 차원은 유지하되 크기를 변경

# [2,2,3] => [-1,1,3]로 변경
print(ft.view([-1,1,3]))
print((ft.view([-1,1,3])).size())
# tensor([[[ 0.,  1.,  2.]],

#         [[ 3.,  4.,  5.]],

#         [[ 6.,  7.,  8.]],

#         [[ 9., 10., 11.]]])
# torch.Size([4, 1, 3])



### .squeeze() : 차원이 1인 차원을 제거

ft = torch.FloatTensor([[0,],[1],[2]])
print(ft)
print(ft.shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

print(ft.squeeze()) #tensor([0., 1., 2.])
print(ft.squeeze().shape) #torch.Size([3])


### .unsqueeze : 특정 위치에 1인 차원을 추가
ft = torch.FloatTensor([0,1,2])
print(ft) #tensor([0., 1., 2.])
print(ft.shape) #torch.Size([3])

print(ft.unsqueeze(0)) #tensor([[0., 1., 2.]])
print(ft.unsqueeze(0).shape) #torch.Size([1, 3])

print(ft.view(1,-1)) #tensor([[0., 1., 2.]])
print(ft.view(1,-1).shape) #torch.Size([1, 3])

print(ft.unsqueeze(1))
# tensor([[0.],
#         [1.],
#         [2.]]) 
print(ft.unsqueeze(1).shape) #torch.Size([3, 1])

print(ft.unsqueeze(-1))
# tensor([[0.],
#         [1.],
#         [2.]])
print(ft.unsqueeze(-1).shape) #torch.Size([3, 1])

# 즉, view(),squeeze(),unsqueeze()는 텐서의 '원소의 수'는 그대로 유지하면서 모양과 차원을 조절한다.



### 타입 캐스팅(Type Casting)
# 텐서에는 데이터형별로 자료형이 존재한다.
# ex) 32비트 부동 소수점 == torch.FloatTensor
# ex) 64비트 signed 정수 == torch.LongTensor

# GPU 연산을 위한 자료형은 torch.cuda.FloatTensor 이다.

# 자료형을 변환하는 것을 '타입 캐스팅' 이라고 한다.

lt = torch.LongTensor([1,2,3,4])
print(lt) #tensor([1, 2, 3, 4])

print(lt.float()) #tensor([1., 2., 3., 4.]) #float형으로 형변환

bt = torch.ByteTensor([True,False,True,False])
print(bt) #tensor([1, 0, 1, 0], dtype=torch.uint8)

print(bt.long()) #tensor([1, 0, 1, 0]) #long형으로 형변환
print(bt.float()) #tensor([1., 0., 1., 0.]) #float형으로 형변환



### 연결하기(Concatenate)
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])

# torch.cat() 의 dim 인자에 따라 어느 차원을 늘릴지 결정된다.

print(torch.cat([x,y],dim=0)) # dim=0 즉 '행(세로)' 기준으로 차원이 늘어남
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.],
#         [7., 8.]])

print(torch.cat([x,y],dim=1)) # dim=1 즉 '열(가로)' 기준으로 차원이 늘어남
# tensor([[1., 2., 5., 6.],
#         [3., 4., 7., 8.]])


### 스택킹(Stacking)
# 연결을 하는 또 다른 방법

x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])

print(torch.stack([x,y,z])) # 3개의 벡터가 순차적으로 쌓임
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])

# 위 명령어는 아래와 같다.
print(torch.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)],dim=0))

print(x.shape) #torch.Size([2])

# 원래 x,y,z의 모양은 (2,). .unsqueeze(0)으로 (1,2)의 2차원 텐서로 만들어준다.
# 이후 cat을 사용하면 (3,2) 텐서로 변경!

# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])

print(torch.stack([x,y,z],dim=1))
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])

# ones_like와 zors_like로 1과 0으로 채워지는 텐서

x = torch.FloatTensor([[0,1,2],[3,4,5]])
print(x)
# tensor([[0., 1., 2.],
#         [3., 4., 5.]])

# .ones_like()를 사용하면 '크기'는 동일하지만 '1'로만 채워진 텐서를 생성
print(torch.ones_like(x))
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

# .zeros_like()를 사용하면 '크기'는 동일하지만 '0'으로만 채워진 텐서를 생성
print(torch.zeros_like(x))
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])



### 덮어쓰기 연산(In-place Operation)
x = torch.FloatTensor([[1,2],[3,4]])
print(x.mul(2))
# tensor([[2., 4.],
#         [6., 8.]])
print(x)
# tensor([[1., 2.],
#         [3., 4.]])

# 기존의 x에 덮어쓰지 않았으므로 x 값이 변하지 않는다.

# '연산' 뒤에 '_'을 붙이면 기존의 값에 덮어 씌운다.
print(x.mul_(2))
print(x)
# tensor([[2., 4.],
#         [6., 8.]])