import numpy as np
import matplotlib.pyplot as plt

width = 30
df = 1


def cal_nbhd(ans, center):
    #cal right
    if k != width-1:
        if ans[k + 1] > ans[k] + df:
            ans[k + 1] = ans[k] + df
    #cal left
    if k != 0:
        if ans[k - 1] > ans[k] + df:
            ans[k - 1] = ans[k] + df
    return ans

#変数初期化
ans = np.zeros(width)
rand1 = np.random.randint(0, width)
rand2 = np.random.randint(0, width)
while rand1 == rand2:
    rand2 = np.random.randint(0, width)
for i in range(len(ans)):
    ans[i] = np.inf

#設定
ans[rand1] = 0
ans[rand2] = 0

toRight = True

for i in range(2):
    if toRight:# ->
        toRight = False
        for k in range(width):
            if ans[k] != np.inf:
                asn = cal_nbhd(ans, k)
    else:# <-
        toRight = True
        for k in range(width-1, -1, -1):
            if ans[k] != np.inf:
                ans = cal_nbhd(ans, k)


plt.plot(ans)
plt.show()