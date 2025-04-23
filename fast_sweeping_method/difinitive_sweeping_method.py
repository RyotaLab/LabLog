import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm

N = 128
width = N + 1
height = N + 1
h = 1 / N

core = [0.5,0.5]
r = 0.3

def calculateT(i, j, x):
    c = h

    a = x[i-1, j]
    b = x[i, j-1]

    if a > b:
        a, b = b, a
    
    if b >= a + c:
        return min(a + c, x[i,j])
    else:
        return min(((a+b)/2 + (2*(c**2) - (a - b)**2)**(1/2) / 2), x[i,j])

def distance(c, x):#距離計算
    return ((x[0]-c[0])**2 + (x[1]-c[1])**2)**(1/2)
    
#初期化
ans = np.zeros((width, height))
ans[:] = np.inf

phi = np.zeros((width, height))

for i in range(width):
    for k in range(height):
        phi[i,k] = distance(core, [h*i, h*k]) - r

#初期化
for i in range(width-1):
    for k in range(height-1):
        if phi[i,k] < 0 and phi[i+1,k] < 0 and phi[i,k+1] < 0 and phi[i+1,k+1] < 0:
            pass
        elif phi[i,k] > 0 and phi[i+1,k] > 0 and phi[i,k+1] > 0 and phi[i+1,k+1] > 0:
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
                    ans[i+s, k+t] = min(ans[i+s,k+t], abs((phi[i+s,k+t])/norm))

#計算
for si in [-1, 1]:
    for sj in [-1, 1]:
        x = ans[::si, ::sj]

        for i in range(1, width):
            for j in range(1, height):
                x[i,j] = calculateT(i, j, x)

# fix the sign
ans[phi < 0] *= -1

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
norm = TwoSlopeNorm(vmin=ans.min(), vcenter=0, vmax=ans.max())

plt.imshow(ans, cmap='coolwarm', norm=norm)  # seismic は青-白-赤の色分け
#plt.imshow(ans, cmap='coolwarm')  # seismic は青-白-赤の色分け
plt.colorbar()
plt.show()