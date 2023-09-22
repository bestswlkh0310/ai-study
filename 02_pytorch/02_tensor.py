import torch
import numpy as np

print(" - tensor")

# tensor
data = [[1, 2], [3, 4]]
newData = torch.tensor(data)

print(newData)
print(" - tensor <-> np")

# tencor <-> np
arr = [
    [2, 3],
    [4, 5]
]

tcArr = torch.tensor(arr)
npArr = tcArr.numpy()
print(npArr)

print(" - random")

# random
shape = (100, 200,)
randArr = torch.rand(shape)
oneArr = torch.ones(shape)
zeroArr = torch.zeros(shape)

print(randArr)
print(oneArr)
print(zeroArr)

# info

print(" - info")

print(randArr.shape)
print(randArr.dtype)
print(randArr.device)

if torch.cuda.is_available():
    print("ableable")

# slice
tc = torch.ones((3, 3,))
print(tc)
tc[:] = 0
print(tc)

# cat

print(" - cat")

tc1 = torch.ones((3, 4,))
tc2 = torch.ones((3, 10,))

print(torch.cat([tc1, tc2], dim=1)) # 가로로 붙이기

tc3 = torch.rand((3, 10,))
tc4 = torch.rand((9, 10,))

print(torch.cat([tc3, tc4])) # 세로로 붙이기

# mul - 단순히 각 요소끼리 곱하는 것

print(" - mul")

tc1 = torch.rand((3, 10,))
tc2 = torch.rand((3, 10,))

print(tc1 * tc2 * 10)

# matmul - 행렬곱

print(" - matmul")

tc1 = torch.rand((3, 10,))
tc2 = torch.rand((7, 10,))

print(tc1 @ tc2.T)

