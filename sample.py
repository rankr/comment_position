#coding: utf-8

import random as rd
import pandas as pd

#sample from 49836-1 commment-code pairs


s = set()
i = 0
while 1:
	k = rd.randint(2, 49836)
	if k not in s:
		s.add(k)
		i += 1
		if i == 300:
			break
a = list(s)
a.sort()
with open('sample.txt', 'w') as w:
	for i in a:
		w.write("%d\n"%i)



a = pd.read_csv('comment_position.csv')
res = []
with open('sample.txt') as f:
    d = f.readlines()
    d = [int(i.strip()) for i in d]
    res = a.loc[d]

res.to_csv('sample.csv')