import copy
import numpy as np
import random
import matplotlib.pyplot as plt
import time

#Variable Definition
alpha = 0.1
lam = 100
gamma = 0.3
tau = 0.12 #0.99 / 8
mu = 0.1
width = 20
height = 20

np.random.seed(1)

# x：二次元 -> D(x)：三次元
def D(x):
    Dx = np.zeros((height, width, 2))
    for n1 in range(height):
        for n2 in range(width):
            if n1 != height-1:
                Dx[n1, n2, 0] = x[n1+1, n2] - x[n1, n2]
            if n2 != width-1:
                Dx[n1, n2, 1] = x[n1, n2+1] - x[n1, n2]
    return Dx

# u：三次元 -> trans_D(u)：二次元
def trans_D(u):
    tDu = np.zeros((height, width))
    for n1 in range(height):
        for n2 in range(width):
            t = 0
            if n1 < height - 1:
                t += -u[n1, n2, 0]
            if n1 >= 1:
                t += u[n1-1, n2, 0]
            if n2 < width - 1:
                t += -u[n1, n2, 1]
            if n2 >= 1:
                t += u[n1, n2-1, 1]
            tDu[n1, n2] = t
    return tDu

# v：二次元 * 3　-> u：三次元
def C(v1, v2, v3):
    C_v1v2v3 = np.zeros((height, width, 2))
    for n1 in range(height):
        for n2 in range(width):
            t1 = 0
            t1 += v1[n1, n2, 0]
            tmp = v2[n1, n2, 0]
            if n1 < height - 1 and n2 > 0:
                tmp += v2[n1+1, n2-1, 0]
            if n1 < height -1:
                tmp += v2[n1+1, n2 ,0]
            if n2 > 0:
                tmp += v2[n1, n2-1, 0]
            t1 += tmp / 4
            tmp = v3[n1, n2, 0]
            if n1 < height -1:
                tmp += v3[n1+1, n2, 0]
            t1 += tmp / 2
            C_v1v2v3[n1, n2, 0] = t1

            t2 = 0
            tmp = v1[n1, n2, 1]
            if n1 > 0 and n2 < width -1:
                tmp += v1[n1-1, n2+1, 1]
            if n1 > 0:
                tmp += v1[n1-1, n2, 1]
            if n2 < width -1:
                tmp += v1[n1, n2+1, 1]
            t2 += tmp / 4
            t2 += v2[n1, n2, 1]
            tmp = v3[n1, n2, 1]
            if n2 < width -1:
                tmp += v3[n1, n2+1, 1]
            t2 += tmp / 2
            C_v1v2v3[n1, n2, 1] = t2
    return -C_v1v2v3

# u：三次元 -> v：二次元 * 3
def trans_C(u):
    v1 = np.zeros((height, width, 2))
    v2 = np.zeros((height, width, 2))
    v3 = np.zeros((height, width, 2))

    for n1 in range(height):
        for n2 in range(width):
            t = 0
            v1[n1, n2, 0] = u[n1, n2, 0]
            t = u[n1, n2, 1]
            if n1 < height -1 and n2 > 0:
                t += u[n1+1, n2-1, 1]
            if n1 < height -1:
                t += u[n1+1, n2, 1]
            if n2 > 0:
                t += u[n1, n2-1, 1]
            v1[n1, n2 ,1] = t / 4

    for n1 in range(height):
        for n2 in range(width):
            t = 0
            t = u[n1, n2, 0]
            if n1 > 0 and n2 < width - 1:
                t += u[n1-1, n2+1, 0]
            if n1 > 0:
                t += u[n1-1, n2, 0]
            if n2 < width -1:
                t += u[n1, n2+1, 0]
            v2[n1, n2, 0] = t / 4
            v2[n1, n2, 1] = u[n1, n2, 1]
    
    for n1 in range(height):
        for n2 in range(width):
            t = 0
            t = u[n1, n2, 0]
            if n1 > 0:
                t += u[n1-1, n2, 0]
            v3[n1, n2, 0] = t / 2
            t = u[n1, n2, 1]
            if n2 > 0:
                t += u[n1, n2-1, 1]
            v3[n1, n2, 1] = t / 2
    return -v1, -v2, -v3


def proxF(x, y):
    proxFx = np.zeros((height, width))
    for n1 in range(height):
        for n2 in range(width):
            proxFx[n1, n2] = (x[n1, n2] + alpha * y[n1, n2]) / (1 + alpha)
    return proxFx

# v↕︎ or v↔︎ or v.
def proxG(v):
    proxGv = np.zeros((height, width, 2))
    for n1 in range(height):
        for n2 in range(width):
            proxGv[n1, n2, 0] = v[n1, n2, 0] - (v[n1, n2, 0] / max(abs(v[n1, n2, 0] / (alpha * lam)), 1))
            proxGv[n1, n2, 1] = v[n1, n2, 1] - (v[n1, n2, 1] / max(abs(v[n1, n2, 1] / (alpha * lam)), 1))
    return proxGv
    

#Noisy image definition y[height, width]
y = np.zeros((height, width), dtype=int)
for i in range(height):
    for k in range(width):
        if i + k < (height + width) / 2:
            y[i][k] = 256
        tmp = random.randint(1, 100)
        if tmp < 30:
            y[i][k] = random.randint(0, 256)
#x = copy.copy(y)
x = np.zeros((height, width))
u = np.zeros((height, width, 2))
v1 = np.zeros((height, width, 2))
v2 = np.zeros((height, width, 2))
v3 = np.zeros((height, width, 2))

for n1 in range(height):
    for n2 in range(width):
        x[n1, n2] = random.randint(1, 10)
        v1[n1, n2] = random.randint(1, 10)
        v2[n1, n2] = random.randint(1, 10)
        v3[n1, n2] = random.randint(1, 10)
        for i in range(2):
            u[n1, n2, i] = random.randint(1, 10)



diff = 10
count = 0


start = time.time()
while diff > 1e-4:
    old_u = u
    count +=1
    x = proxF(x - tau*trans_D(D(x) + C(v1, v2, v3) + mu*u), y)

    t1, t2, t3 = trans_C(D(x) + C(v1, v2, v3) + mu*u)
    v1 = proxG(v1 - gamma*t1)
    v2 = proxG(v2 - gamma*t2)
    v3 = proxG(v3 - gamma*t3)

    u = u + (D(x) + C(v1, v2, v3)) / mu

    diff = np.sum(np.abs(old_u - u))
    # if count > 500:
    #     break
end = time.time()
print(end - start)

np.savetxt("2d_ydata.txt", y)
np.savetxt("2d_xdata.txt", x)

print("count =", count)
print(np.sum(x - y))

plt.subplot(1, 2, 1)
plt.imshow(y, cmap='viridis')  # ヒートマップの色を指定
plt.colorbar()                    # カラーバーを表示
plt.title("y")  # タイトル

plt.subplot(1, 2, 2)
plt.imshow(x, cmap='viridis')  # ヒートマップの色を指定
plt.colorbar()                    # カラーバーを表示
plt.title("x")  # タイトル
plt.tight_layout()
plt.show()