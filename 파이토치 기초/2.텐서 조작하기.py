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
