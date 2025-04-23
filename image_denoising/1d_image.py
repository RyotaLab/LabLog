import numpy as np
import copy
import matplotlib.pyplot as plt

#Variable Definition
alpha = 0.1
lam = 10.0
gamma = 0.1
tau = 0.99/8 #0.99 / 8
mu = 0.1
width = 100

#x：1次元
def proxF(x, y):
    proxFx = np.zeros((width))
    for n1 in range(width):
        proxFx[n1] = (x[n1] + alpha * y[n1]) / (1 + alpha)
    return proxFx
    
#v：1次元
def proxG(v):
    proxGv = np.zeros((len(v)))
    for n1 in range(len(v)):
        proxGv[n1] = v[n1] - (v[n1] / max(abs(v[n1]) / (alpha*lam), 1))
    return proxGv

#x：1次元　-> u：1次元
def D(x):
    Dx = np.zeros((width))
    for n1 in range(width):
        if n1 < width - 1:
            Dx[n1] = x[n1+1] - x[n1]
    return Dx

#u：1次元　->　x：1次元
def trans_D(u):
    tDu = np.zeros((width))
    for n1 in range(width):
        if n1 < width -1:
            tDu[n1] = -u[n1]
        if n1 > 0:
            tDu[n1] += u[n1-1]
    return tDu

#v：1次元 * 2　-> u：1次元
def C(v1, v2):
    tLv = np.zeros((width))
    for n1 in range(width-1):
        tLv[n1] = v1[n1]
        tmp = v2[n1] + v2[n1+1]
        tLv[n1] += tmp / 2
    return -tLv

#u：1次元　->　v：1次元 * 2
def trans_C(u):
    v1 = np.zeros((width))
    v2 = np.zeros((width))
    v2[0] = u[0] / 2
    v2[width-1] = u[width-2] / 2
    for n1 in range(width-1):
        v1[n1] = u[n1]
        if n1 > 0:
            v2[n1] = (u[n1] + u[n1-1]) / 2
    return -v1, -v2


y = np.zeros((width))
for i in range(width):
    if i > 30 and i < width - 30:
        y[i] = 30.

x = copy.copy(y)
u = np.zeros((width))
v1 = np.zeros((width))
v2 = np.zeros((width))

print(x)

v1 = np.random.rand(width)
v2 = np.random.rand(width)
u = np.random.rand(width)

#trans_C(u) * v1, v2 == u * C(v1, v2)
t1, t2 = trans_C(u)
print((t1 @ v1) + (t2 @ v2))
print(u @ C(v1, v2))
#D(x) * u == x * trans_D(u)
print(D(x) @ u)
print(x @ trans_D(u))

diff_L = []
diff = 10
count = 0
while diff > 1e-3:
    if count < 10:
        print(x)
    count += 1
    old_u = u
    x = proxF(x - tau*trans_D(D(x) + C(v1, v2) + mu*u), y)
    t1, t2 = trans_C(D(x) + C(v1, v2) + mu*u)
    v1 = proxG(v1 - gamma*t1)
    v2 = proxG(v2 - gamma*t2)
    u = u + ((D(x) + C(v1, v2)) / mu)
    
    diff = abs(np.sum(old_u - u))
    diff_L.append(diff)

    # if count > 2000:
    #     break

label = range(len(y)) 

np.savetxt("1d_ydata.txt", y)
np.savetxt("1d_xdata.txt", x)

plt.plot(label, x, color='blue', label="oparated")
plt.plot(label, y, color='green', label="original")

# diff_L_label = range(len(diff_L)) 
# plt.plot(diff_L_label, diff_L, color='red', label="diff")

plt.show()