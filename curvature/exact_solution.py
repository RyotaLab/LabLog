import numpy as np
import matplotlib.pyplot as plt

N = 128
width = N+1
height = N+1
center = [0.5, 0.5]
h = 1 / N
lam = 0.001
rho = np.sqrt(3 * np.pi * lam)
r = 0.4

true_ans = np.zeros((width, height))

def cal_dis(center, point):
    return np.sqrt(abs(center[0]-point[0])**2 + abs(center[1]-point[1])**2)

for i in range(width):
    for k in range(height):
        absX = cal_dis(center, [i*h, k*h])
        if absX > rho:
            true_ans[i, k] = absX - r + (lam / absX)
        else:
            true_ans[i, k] = rho - r + (lam / rho)

#loop そもそもrの更新式でルートの中が負になる
for index in range(1000):
    for i in range(width):
        for k in range(height):
            absX = cal_dis(center, [i*h, k*h])
            if absX > rho:
                true_ans[i, k] = absX - r + (lam / absX)
            else:
                true_ans[i, k] = rho - r + (lam / rho)
    if r**2 / 4 - lam < 0:
        print(index)
        break
    r = r / 2 + np.sqrt(r**2 / 4 - lam)



a = np.linspace(0.0, 1.0, height)
b = np.linspace(0.0, 1.0, width)
a,b = np.meshgrid(a,b)
fig = plt.figure()
ax2 = plt.subplot(projection='3d')
ax2.plot_surface(a, b, true_ans)
plt.show()