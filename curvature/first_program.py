from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

#Variable Definition

N = 256
width = N + 1
height = N + 1
h = 1 / N
core = [0.5,0.5]
r = 0.4

lam = h / 10
gamma = 0.3
tau = 0.99/8 * h**2
mu = 256 /h

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

# x：二次元 -> D(x)：三次元
def D(x):
    Dx = np.zeros((height, width, 2))
    Dx[:-1,:,0] = (x[1:,:] - x[:-1,:])
    Dx[:,:-1,1] = (x[:,1:] - x[:,:-1])
    return Dx / h

# u：三次元 -> trans_D(u)：二次元
def trans_D(u):
    tDu = np.zeros((height, width))
    tDu[:,:] = -u[:,:,0]
    tDu[1:,:] += u[:-1,:,0]
    tDu[:,:] += -u[:,:,1]
    tDu[:,1:] += u[:,:-1,1]
    return tDu / h

# v：二次元 * 3　-> u：三次元
def C(v1, v2, v3):
    C_v1v2v3 = np.zeros((height, width, 2))
    tmp = np.zeros((height,width))
    C_v1v2v3[:,:,0] = v1[:,:,0]
    tmp[:,:] = v2[:,:,0]
    tmp[:-1,1:] += v2[1:,:-1,0]
    tmp[:-1,:] += v2[1:,:,0]
    tmp[:,1:] += v2[:,:-1,0]
    C_v1v2v3[:,:,0] += (tmp * 0.25)
    tmp[:,:] = v3[:,:,0]
    tmp[:-1,:] += v3[1:,:,0]
    C_v1v2v3[:,:,0] += (tmp * 0.5)

    tmp[:,:] = v1[:,:,1]
    tmp[1:,:-1] += v1[:-1,1:,1]
    tmp[1:,:] += v1[:-1,:,1]
    tmp[:,:-1] += v1[:,1:,1]
    C_v1v2v3[:,:,1] = (tmp * 0.25)
    C_v1v2v3[:,:,1] += v2[:,:,1]
    tmp[:,:] = v3[:,:,1]
    tmp[:,:-1] = v3[:,1:,1]
    C_v1v2v3[:,:,1] += (tmp * 0.5)

    return -C_v1v2v3

# u：三次元 -> v：二次元 * 3
def trans_C(u):
    v1 = np.zeros((height, width, 2))
    v2 = np.zeros((height, width, 2))
    v3 = np.zeros((height, width, 2))

    v1[:,:,0] = u[:,:,0]
    v1[:,:,1] = u[:,:,1]
    v1[:-1,1:,1] += u[1:,:-1,1]
    v1[:-1,:,1] += u[1:,:,1]
    v1[:,:-1,1] += u[:,1:,1]
    v1[:,:,1] *= 0.25

    v2[:,:,0] = u[:,:,0]
    v2[1:,:-1,0] += u[:-1,1:,0]
    v2[1:,:,0] += u[:-1,:,0]
    v2[:,:-1,0] += u[:,1:,0]
    v2[:,:,0] *= 0.25
    v2[:,:,1] = u[:,:,1]

    v3[:,:,0] = u[:,:,0]
    v3[1:,:,0] = u[:-1,:,0]
    v3[:,:,0] *= 0.5
    v3[:,:,1] = u[:,:,1]
    v3[:,1:,1] += u[:,:-1,1]
    v3[:,:,1] *= 0.5
    
    return -v1, -v2, -v3

def proxF(x, y, alpha):
    return (x + alpha * y) / (1 + alpha)

# v↕︎ or v↔︎ or v.
def proxG(v, alpha):
    w = np.sqrt(v[:,:,0]**2 + v[:,:,1]**2) [:, :, None]
    proxGv = v - (v / np.maximum(w / (alpha * lam), 1))
    return proxGv

def cal_diff(approx, exact):
    max_diff = 0
    maxpoint = [0,0]
    for i in range(width):
        for k in range(height):
            if i > N/3 and i < 2*N/3 and k > N/3 and k < 2*N/3:
                diff = abs(approx[i,k]-exact[i,k])
                max_diff = max(max_diff, diff)
                if max_diff == diff:
                    maxpoint = [i,k]
    return max_diff, maxpoint

#Exact soltion
true_ans = np.zeros((width, height))
rho = np.sqrt(np.pi * lam)
true_ans = np.zeros((width, height))
def cal_dis(center, point):
    return np.sqrt(abs(center[0]-point[0])**2 + abs(center[1]-point[1])**2)
for i in range(width):
    for k in range(height):
        absX = cal_dis(core, [i*h, k*h])
        if absX > rho:
            true_ans[i, k] = absX - r + (lam / absX)
        else:
            true_ans[i, k] = rho - r + (lam / rho)
#Exact sol end

#用意した画像
y = np.zeros((height, width))
x = np.zeros((height, width))
u = np.zeros((height, width, 2))
v1 = np.zeros((height, width, 2))
v2 = np.zeros((height, width, 2))
v3 = np.zeros((height, width, 2))

# for n1 in range(height):
#     for n2 in range(width):
#         x[n1, n2] = np.random.randint(1, 10)
#         v1[n1, n2] = np.random.randint(1, 10)
#         v2[n1, n2] = np.random.randint(1, 10)
#         v3[n1, n2] = np.random.randint(1, 10)
#         for i in range(2):
#             u[n1, n2, i] = np.random.randint(1, 10)
count = 0

ans = np.zeros((width, height))
ans[:] = np.inf

initial_image = np.zeros((width, height))

true_point = 0

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

# plt.imshow(ans)
# plt.show()

#距離計算（sweep）
for si in [-1, 1]:
    for sj in [-1, 1]:
        tmp = ans[::si, ::sj]
        for i in range(1, width):
            for j in range(1, height):
                tmp[i,j] = calculateT(i, j, tmp)

# fix the sign
ans[phi < 0] *= -1

initial_image = ans.copy()

#loop
for index in range(3):

    y = ans

    count = 0
    while True: #diff > 1e-2:

        old_u = u
        count +=1
        x = proxF(x - tau*trans_D(D(x) + C(v1, v2, v3) + mu*u), y, tau*mu)

        t1, t2, t3 = trans_C(D(x) + C(v1, v2, v3) + mu*u)
        v1 = proxG(v1 - gamma*t1, gamma*mu)
        v2 = proxG(v2 - gamma*t2, gamma*mu)
        v3 = proxG(v3 - gamma*t3, gamma*mu)

        u = u + (D(x) + C(v1, v2, v3)) / mu

        diff = np.sum(np.abs(old_u - u))
        if count >= 64:
            break

    # #　表示
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax1.imshow(initial_image)
    # ax2.imshow(x)
    # ax1.set_title("initial")
    # ax2.set_title(f'graph{index}')
    # plt.savefig(f"graph_{index}")
    # print(f'save graph{index}')

    if index == 0:
    #     a = np.linspace(0.0, 1.0, height)
    #     b = np.linspace(0.0, 1.0, width)
    #     a,b = np.meshgrid(a,b)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1,1,1, projection='3d')
    #     ax.plot_surface(a, b, x-true_ans)

        print(cal_diff(x, true_ans))

    ans[:] = np.inf

    #次の初期値決め
    for i in range(width-1):
        for k in range(height-1):
            if x[i,k] < 0 and x[i+1,k] < 0 and x[i,k+1] < 0 and x[i+1,k+1] < 0:
                pass
            elif x[i,k] > 0 and x[i+1,k] > 0 and x[i,k+1] > 0 and x[i+1,k+1] > 0:
                pass
            else:#初期値をおく
                dw1 = x[i,k] - x[i+1,k]
                dw2 = x[i,k+1] - x[i+1,k+1]
                dw = (dw1+dw2)/(2 * h)
                dh1 = x[i,k] - x[i,k+1]
                dh2 = x[i+1,k] - x[i+1,k+1]
                dh = (dh1+dh2)/(2 * h)
                norm = (dw**2 + dh**2) ** (1/2)
                for s in range(2):
                    for t in range(2):
                        ans[i+s, k+t] = min(ans[i+s,k+t], abs((x[i+s,k+t])/norm))


    #距離計算（sweep）
    for si in [-1, 1]:
        for sj in [-1, 1]:
            tmp = ans[::si, ::sj]
            for i in range(1, width):
                for j in range(1, height):
                    tmp[i,j] = calculateT(i, j, tmp)


    ans[x < 0] *= -1
    # fix the sign
