import re
import os
import pandas as pd
import math

import func
import extract_comment_position as ecp
import complexity_measure as cm

cmt1 = re.compile("\s*/\*")
cmt2 = re.compile("\s*//")


def code_line_num(string):
    string = string.replace("[enter]", '\n')
    string = string.split('\n')
    ret = 0
    in_comment = 0
    for i in string:
        if not in_comment:
            if re.match(cmt1, i):
                if '*/' not in i:
                    in_comment = 1
            elif '/*' in i:
                ret += 1
                if "*/" not in i:
                    in_comment += 1
            elif re.match(cmt2, i):
                continue
            else:
                ret += 1
    return ret

c = pd.read_csv("/Users/apple/Documents/work1/code/result/linux_func_with_comment_stat.csv")
n = pd.read_csv("/Users/apple/Documents/work1/code/result/linux_func_without_comment_stat.csv")



a = max(c['line_num']) #1829 for linux
b = max(n['line_num']) #2907 for linux
a = max(a, b)



#'''
L = int(math.ceil(a/10))


c_op_num = [0] * L
for i in c['line_num']:
	c_op_num[min(int(i/10), L)] += 1

n_op_num = [0] * L
for i in n['line_num']:
	n_op_num[min(int(i/10), L)] += 1


for c, i, j in zip(range(0, L), c_op_num, n_op_num):
	if j!=0:
		print(c,":\t",i,j,i/(i+j))
import numpy as np
import matplotlib.pyplot as plt
#'''
