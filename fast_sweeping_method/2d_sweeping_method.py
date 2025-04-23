import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm

N = 240
width = N + 1
height = N + 1
h = 1 / N


core = [0.5,0.5]
r = 0.3

#center = tupple(i, j)
def cal_nbhd(ans, center):
    i, j = center
    if ans[i, j] == np.inf:
        return ans
    t = 0
    #右
    if i != width - 1:
        t = calculateT((i+1, j), ans)
        if abs(t) < abs(ans[i+1, j]):
            ans[i+1, j] = t
    #左
    if i != 0:
        t = calculateT((i-1, j), ans)
        if abs(t) < abs(ans[i-1, j]):
            ans[i-1, j] = t
    #下
    if j != height - 1:
        t = calculateT((i, j+1), ans)
        if abs(t) < abs(ans[i, j+1]):
            ans[i, j+1] = t
    #上
    if j != 0:
        t = calculateT((i, j-1), ans)
        if abs(t) < abs(ans[i, j-1]):
            ans[i, j-1] = t
    return ans

def calculateT(center, ans):
    i, j = center
    a = np.inf
    b = np.inf
    c = h

    if i != width - 1:
        a = abs(ans[i+1, j])
        
    if i != 0:
        if a > abs(ans[i-1, j]):
            a = abs(ans[i-1, j])

    if j != height - 1:
        b = abs(ans[i, j+1])

    if j != 0:
        if b > abs(ans[i, j-1]):
            b = abs(ans[i, j-1])

    if a > b:
        tmp = a
        a = b
        b = tmp
    if b > a + c:
        return (a + c)
    else:
        return (((a+b)/2) + ((2*(c**2) - (a - b)**2)**(1/2)) / 2)

def distance(c, x):#距離計算
    return ((x[0]-c[0])**2 + (x[1]-c[1])**2)**(1/2)

    
#初期化
ans = np.zeros((width, height))
ans[:] = np.inf

phi = np.zeros((width, height))

for i in range(width):
    for k in range(height):
        phi[i,k] = distance(core, [h*i, h*k])

#初期化
for i in range(width-1):
    for k in range(height-1):
        if phi[i,k] < r and phi[i+1,k] < r and phi[i,k+1] < r and phi[i+1,k+1] < r:
            pass
        elif phi[i,k] > r and phi[i+1,k] > r and phi[i,k+1] > r and phi[i+1,k+1] > r:
            pass
        else:#初期値をおく
            dw1 = phi[i,k] - phi[i+1,k]
            dw2 = phi[i,k+1] - phi[i+1,k+1]
            dw = (dw1+dw2)/(2 * h)
            dh1 = phi[i,k] - phi[i,k+1]
            dh2 = phi[i+1,k] - phi[i+1,k+1]
            dh = (dh1+dh2)/(2 * h)
            norm = (dw**2 + dh**2) ** (1/2)
            for s in range(2):
                for t in range(2):
                    ans[i+s, k+t] = min(ans[i+s,k+t], abs((phi[i+s,k+t]-r)/norm))

#計算
for _ in range(1):
    #左上から右下
    for i in range(height + width - 1):
        if i < width:
            start = (0, i)
        else:
            start = (i - width + 1, width - 1)
        ans = cal_nbhd(ans, start)
        while True:
            if start[0] == height-1 or start[1] == 0:
                break
            start = (start[0]+1, start[1]-1)
            ans = cal_nbhd(ans, start)
    #右上から左下
    for i in range(height + width - 1):
        if i < height:
            start = (i, width-1)
        else:
            start = (height-1, width - i + height -2)
        ans = cal_nbhd(ans, start)
        while True:
            if start[0] == 0 or start[1] == 0:
                break
            start = (start[0]-1, start[1]-1)
            ans = cal_nbhd(ans, start)
    #右下から左上
    for i in range(height + width - 1):
        if i < width:
            start = (height-1, width-1-i)
        else:
            start = (height -i + width - 2, 0)
        ans = cal_nbhd(ans, start)
        while True:
            if start[0] == 0 or start[1] == width-1:
                break
            start = (start[0]-1, start[1]+1)
            ans = cal_nbhd(ans, start)
    #左下から右上
    for i in range(height + width - 1):
        if i < height:
            start = (height-1-i, 0)
        else:
            start = (0, i - height + 1)
        ans = cal_nbhd(ans, start)
        while True:
            if start[0] == height-1 or start[1] == width-1:
                break
            start = (start[0]+1, start[1]+1)
            ans = cal_nbhd(ans, start)

# fix the sign
ans[phi < r] *= -1

#誤差調べ
max_diff = -np.inf
true_value = 0
for i in range(width):
    for k in range(height):
        if abs(distance(core, (i*h,k*h)) - r) < (h * 5):
            true_value = distance(core, (i*h,k*h)) - r
            max_diff = max(max_diff, abs(true_value - ans[i,k]))


print("max_diff=",max_diff)

#plot
#norm = TwoSlopeNorm(vmin=ans.min(), vcenter=0, vmax=ans.max())

#plt.imshow(ans, cmap='coolwarm', norm=norm)  # seismic は青-白-赤の色分け
plt.imshow(ans, cmap='coolwarm')  # seismic は青-白-赤の色分け
plt.colorbar()
plt.show()