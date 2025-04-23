import numpy as np
import matplotlib.pyplot as plt

width = 20
height = 20

#center = tupple(i, j)
def add_narrowBand(center, ans, Band):
    i, j = center
    t = 0
    #右
    if i != width - 1:#右端でない
        t = calculateT((i+1, j), ans)
        result = next((tmp for tmp in Band if tmp[1] == (i+1, j)), None)
        if result == None and ans[i+1][j] == np.inf:#元々Bandにない&aliveでない
            Band.add((t, (i+1, j)))
        if result != None:
            if result[0] > t:#最小値比較
                Band.remove(result)
                Band.add((t, (i+1, j)))
    #左
    if i != 0:
        t = calculateT((i-1, j), ans)
        result = next((tmp for tmp in Band if tmp[1] == (i-1, j)), None)
        if result == None and ans[i-1][j] == np.inf:#元々Bandにない
            Band.add((t, (i-1, j)))
        if result != None:
            if result[0] > t:#最小値比較
                Band.remove(result)
                Band.add((t, (i-1, j)))
    #下
    if j != height - 1:
        t = calculateT((i, j+1), ans)
        result = next((tmp for tmp in Band if tmp[1] == (i, j+1)), None)
        if result == None and ans[i][j+1] == np.inf:#元々Bandにない
            Band.add((t, (i, j+1)))
        if result != None:
            if result[0] > t:#最小値比較
                Band.remove(result)
                Band.add((t, (i, j+1)))
    #上
    if j != 0:
        t = calculateT((i, j-1), ans)
        result = next((tmp for tmp in Band if tmp[1] == (i, j-1)), None)
        if result == None and ans[i][j-1] == np.inf:#元々Bandにない
            Band.add((t, (i, j-1)))
        if result != None:
            if result[0] > t:#最小値比較
                Band.remove(result)
                Band.add((t, (i, j-1)))
    return Band

def calculateT(center, ans):
    i, j = center
    a = np.inf
    b = np.inf
    c = 1

    if i != width - 1:
        a = ans[i+1][j]
        
    if i != 0:
        if a > ans[i-1][j]:
            a = ans[i-1][j]

    if j != height - 1:
        b = ans[i][j+1]

    if j != 0:
        if b > ans[i][j-1]:
            b = ans[i][j-1]

    if a > b:
        tmp = a
        a = b
        b = tmp
    if b > a + c:
        return a + c
    else:
        return ((a+b)/2) + ((2*(c**2) - (a - b)**2)**(1/2)) / 2

ans = np.zeros((width, height))
rand1 = [np.random.randint(0, width), np.random.randint(0, height)]
rand2 = [np.random.randint(0, width), np.random.randint(0, height)]
# rand1 = [3,3]
# rand2 = [3,2]
narrowBand = set()# (T, (i,j))
for i in range(width):
    for k in range(height):
        ans[i,k] = np.inf

ans[rand1[0]][rand1[1]] = 0
ans[rand2[0]][rand2[1]] = 0
narrowBand = add_narrowBand((rand1[0], rand1[1]), ans, narrowBand)
narrowBand = add_narrowBand((rand2[0], rand2[1]), ans, narrowBand)

count = 0
while len(narrowBand) != 0:
    print(narrowBand)
    minimum = min(narrowBand, key=lambda x: x[0])
    narrowBand.remove(minimum)
    print(count, "minimum = ",minimum)
    count += 1
    print()
    i,j = minimum[1]#(i, j)
    ans[i][j] = minimum[0]
    narrowBand = add_narrowBand((i,j), ans, narrowBand)

print(ans)



plt.imshow(ans, cmap='viridis')  # ヒートマップの色を指定
plt.colorbar()
plt.show()