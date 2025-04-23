'''1d fast marching method
Ω = [0, 20] Grid
Γ = {?, ?} Two Points chosen at random

Γからの距離Tをそれぞれの点に対して求める
'''
import numpy as np
import matplotlib.pyplot as plt

width = 30

def add_priorityQ(ans, point, priorityQ):
    if point != 0 and ans[point-1] == np.inf:
        target = point-1
        exists = any(t[0] == target for t in priorityQ)
        if not exists:
            priorityQ.append((point - 1, ans[point] + 1))
    if point != width-1 and ans[point+1] == np.inf:
        target = point+1
        exists = any(t[0] == target for t in priorityQ)
        if not exists:
            priorityQ.append((point + 1, ans[point] + 1))
    priorityQ = sorted(priorityQ, reverse=False, key=lambda x: x[1])
    return priorityQ
        
#変数初期化
ans = np.zeros(width)
rand1 = np.random.randint(0, width)
rand2 = np.random.randint(0, width)
while rand1 == rand2:
    rand2 = np.random.randint(0, width)
priorityQ = []
for i in range(len(ans)):
    ans[i] = np.inf

#設定
ans[rand1] = 0
ans[rand2] = 0
priorityQ = add_priorityQ(ans, rand1, priorityQ)
priorityQ = add_priorityQ(ans, rand2, priorityQ)

#回していく
while len(priorityQ) > 0:
    point = priorityQ.pop(0)
    ans[point[0]] = point[1]
    priorityQ = add_priorityQ(ans, point[0], priorityQ)
plt.plot(ans)
plt.show()
