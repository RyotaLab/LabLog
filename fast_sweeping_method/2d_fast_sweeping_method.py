import numpy as np
import matplotlib.pyplot as plt

width = 30
height = 30

#center = tupple(i, j)
def cal_nbhd(ans, center):
    i, j = center
    if ans[i, j] == np.inf:
        return ans
    t = 0
    #右
    if i != width - 1:
        t = calculateT((i+1, j), ans)
        if t < ans[i+1, j]:
            ans[i+1, j] = t
    #左
    if i != 0:
        t = calculateT((i-1, j), ans)
        if t < ans[i-1, j]:
            ans[i-1, j] = t
    #下
    if j != height - 1:
        t = calculateT((i, j+1), ans)
        if t < ans[i, j+1]:
            ans[i, j+1] = t
    #上
    if j != 0:
        t = calculateT((i, j-1), ans)
        if t < ans[i, j-1]:
            ans[i, j-1] = t
    return ans

def calculateT(center, ans):
    i, j = center
    a = np.inf
    b = np.inf
    c = 1

    if i != width - 1:
        a = ans[i+1, j]
        
    if i != 0:
        if a > ans[i-1, j]:
            a = ans[i-1, j]

    if j != height - 1:
        b = ans[i, j+1]

    if j != 0:
        if b > ans[i, j-1]:
            b = ans[i, j-1]

    if a > b:
        tmp = a
        a = b
        b = tmp
    if b > a + c:
        return a + c
    else:
        return ((a+b)/2) + ((2*(c**2) - (a - b)**2)**(1/2)) / 2
    
#初期化
ans = np.zeros((width, height))
rand1 = [np.random.randint(0, width), np.random.randint(0, height)]
rand2 = [np.random.randint(0, width), np.random.randint(0, height)]
for i in range(width):
    for k in range(height):
        ans[i,k] = np.inf

ans[rand1[0], rand1[1]] = 0
ans[rand2[0], rand2[1]] = 0


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
        
plt.imshow(ans, cmap='viridis')  # ヒートマップの色を指定
plt.colorbar()
plt.show()


#真解と近似解の誤差を求める