#coding: utf-8

import re
import os
import pandas as pd

import func
import extract_comment_position as ecp
import complexity_measure as cm

if __name__ == "__main__":
	c = pd.read_csv("./result/opencv_func_without_comment.csv")
	w = open('./result/opencv_func_without_comment_stat.csv', 'w')
	w.write("line_num,max_depth,variable_total_num,operator_total_num,op_line_rate\n")
	for code in c['func']:
		#filter out comments in code first
		code = re.sub('\[comma\]', ',', code)
		code = re.sub('\[enter\]', '\n', code)
		snippets, lines_cnt = ecp.code_str_split(code)
		labels = ecp.simple_label(snippets)
		code = ''
		for i, j in zip(snippets, labels):
			if j != 'comment':
				code += i
		#print(code)
		#exit()
		r = cm.complexity_measure(code)
		w.write("%d,%d,%d,%d,%f\n"%(r[0],r[1],r[2],r[3],r[4]))
	w.close()
