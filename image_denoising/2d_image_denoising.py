from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


#Variable Definition
lam = 50
gamma = 0.3
tau = 0.99/8 #0.99 / 8
mu = 5
width = 100
height = 100

# np.random.seed(1)

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


def proxF(x, y, alpha):
    return (x + alpha * y) / (1 + alpha)

# v↕︎ or v↔︎ or v.
def proxG(v, alpha):
    w = np.sqrt(v[:,:,0]**2 + v[:,:,1]**2)
    w = np.reshape(w, (100, 100, 1))
    proxGv = v - (v / np.maximum(w / (alpha * lam), 1))

    return proxGv

#用意した画像
image_file = "/Users/tanabou/ClassesEtc/seminar/image_denoising/noised_circle.png"
y = np.array(Image.open(image_file))

# a = np.linspace(-0.5, 0.5, height)
# b = np.linspace(-0.5, 0.5, width)
# a,b = np.meshgrid(a,b)
# y = np.sqrt(a**2 + b**2)

x = np.zeros((height, width))
u = np.zeros((height, width, 2))
v1 = np.zeros((height, width, 2))
v2 = np.zeros((height, width, 2))
v3 = np.zeros((height, width, 2))

diff = 10
count = 0

start = time.time()
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
    if count >= 3000:
        break

end = time.time()
print("mu =", mu)
print("time =",end - start)
print("diff =", diff)
print("np.sum(x - y) =",np.sum(x - y))
print("count =", count)


plt.imshow(x)
plt.colorbar()
plt.show()

# ax = plt.subplot(projection='3d')
# ax.plot_surface(a, b, x)
# plt.show()

# 保存コマンドーーーー
pil_img = Image.fromarray(x)
pil_img = pil_img.convert("L")
pil_img.save("proposedTV.png")