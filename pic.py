import re
import os
import pandas as pd

import func
import extract_comment_position as ecp
import complexity_measure as cm

c = pd.read_csv("/Users/apple/Documents/work1/code/result/linux_func_with_comment_stat.csv")
n = pd.read_csv("/Users/apple/Documents/work1/code/result/linux_func_without_comment_stat.csv")
    
c_op_num = [0,0,0,0,0,0,0,0,0,0,0]
for i in c['op_line_rate']:
	c_op_num[min(int(i*10), 10)] += 1

n_op_num = [0,0,0,0,0,0,0,0,0,0,0]
for i in n['op_line_rate']:
	n_op_num[min(int(i*10), 10)] += 1



import numpy as np
import matplotlib.pyplot as plt


name_list = []
left = 0
right = 0.1
for i in range(0, 10):
	name_list.append("%.1f-%.1f"%(left, right))
	left += 0.1
	right += 0.1
name_list.append(">1")

print(name_list)


num_list = []
for i in range(0, len(name_list)):
	num_list.append(c_op_num[i]/(c_op_num[i]+n_op_num[i]))


total_width, n = 11, 11
width = total_width / (2*n)
x = list(range(len(num_list)))
print(x)
 
plt.bar(x, num_list, width=width, label='commented_rate',tick_label = name_list, fc = 'y')
plt.legend()
plt.show()
