#coding: utf-8

import json
import functools

func_cnt = json.load(open("/Users/apple/Documents/work1/code/result/linux_func_files_filtered.json"))
h = {}
for func in func_cnt:
    c = func_cnt[func][0]
    if c not in h:
        h[c] = [0, 0]
    if func_cnt[func][1] == 1:
        h[c][0] += 1
    h[c][1] += 1
k = list(h.keys())

k.sort()

x = range(0, 101, 5)
y = [0] * 20
z = [0] * 20
a100 = 0
b100 = 0
for i in k:
    if h[i][1] == 0:
        continue
    if i < 100:
        y[i//5]+=h[i][0]
        z[i//5]+=h[i][1]
    else:
        a100 += h[i][0]
        b100 += h[i][1]

for i in range(0, len(y)):
	y[i] = y[i] / z[i]
y.append(a100/b100)
for i,j,k in zip(x, y, z):
    print(i,j,k)


import matplotlib.pyplot as plt# 导入模块
# 指定列表Y坐标为列表中的值
plt.plot(x,y,linewidth=5)# linewidth决定绘制线条的粗细
plt.title('Func-Annotated-Rate',fontsize=24)# 标题
plt.xlabel('Files Num Appeared',fontsize=14)
plt.ylabel('Annotated Rate',fontsize=14)
plt.tick_params(axis='both',labelsize=14)# 刻度加粗
plt.show()# 输出图像
