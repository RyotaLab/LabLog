from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


#Variable Definition
alpha = 0.1
lam = 0.1
gamma = 0.3
tau = 0.99/8 #0.99 / 8
mu = 5.0

N = 15
width = N+1
height = N+1
h = 1 / N
core = [0.5,0.5]
r = 0.3

def distance(c, x):#距離計算
    return ((x[0]-c[0])**2 + (x[1]-c[1])**2)**(1/2)


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

# x：二次元 -> D(x)：三次元
def D(x):
    Dx = np.zeros((height, width, 2))
    Dx[:-1,:,0] = x[1:,:] - x[:-1,:]
    Dx[:,:-1,1] = x[:,1:] - x[:,:-1]
    return Dx

# u：三次元 -> trans_D(u)：二次元
def trans_D(u):
    tDu = np.zeros((height, width))
    tDu[:,:] = -u[:,:,0]
    tDu[1:,:] += u[:-1,:,0]
    tDu[:,:] += -u[:,:,1]
    tDu[:,1:] += u[:,:-1,1]
    return tDu

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


def proxF(x, y):
    return (x + alpha * y) / (1 + alpha)

# v↕︎ or v↔︎ or v.
def proxG(v):
    proxGv = np.zeros((height, width, 2))
    proxGv[:,:,0] = v[:,:,0] - (v[:,:,0] / np.maximum(np.abs(v[:,:,0] / (alpha * lam)), 1))
    proxGv[:,:,1] = v[:,:,1] - (v[:,:,1] / np.maximum(np.abs(v[:,:,1] / (alpha * lam)), 1))
    return proxGv

#用意した画像
ans = np.zeros((width, height))
initial_image = np.zeros((width, height))
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

#距離計算（sweep）
for si in [-1, 1]:
    for sj in [-1, 1]:
        tmp = ans[::si, ::sj]

        for i in range(1, width):
            for j in range(1, height):
                tmp[i,j] = calculateT(i, j, tmp)

# fix the sign
# ans[phi < 0] *= -1

x = np.zeros((height, width))
u = np.zeros((height, width, 2))
v1 = np.zeros((height, width, 2))
v2 = np.zeros((height, width, 2))
v3 = np.zeros((height, width, 2))

for n1 in range(height):
    for n2 in range(width):
        x[n1, n2] = np.random.randint(1, 10)
        v1[n1, n2] = np.random.randint(1, 10)
        v2[n1, n2] = np.random.randint(1, 10)
        v3[n1, n2] = np.random.randint(1, 10)
        for i in range(2):
            u[n1, n2, i] = np.random.randint(1, 10)

diff = 10
count = 0

# image_file = "/Users/tanabou/ClassesEtc/seminar/image_denoising/noised_circle.png"
# ans = np.array(Image.open(image_file))

while True: #diff > 1e-2:
    old_u = u
    count +=1
    x = proxF(x - tau*trans_D(D(x) + C(v1, v2, v3) + mu*u), ans)

    t1, t2, t3 = trans_C(D(x) + C(v1, v2, v3) + mu*u)
    v1 = proxG(v1 - gamma*t1)
    v2 = proxG(v2 - gamma*t2)
    v3 = proxG(v3 - gamma*t3)

    u = u + (D(x) + C(v1, v2, v3)) / mu

    diff = np.sum(np.abs(old_u - u))
    print(count)
    if count >= 1000:
        break

# plot
plt.imshow(x, cmap='coolwarm')  # seismic は青-白-赤の色分け
plt.colorbar()
plt.show()