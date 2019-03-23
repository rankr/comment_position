#coding: utf-8

import re
import os

import func
import extract_comment_position as ecp

if __name__ == "__main__":
	path = "opencv/modules"
	file_paths = func.cpp_file_select(path)

    if not os.path.exists('result'):
		os.mkdir('result');
	w = open("result/comment_position.csv", 'w', encoding = "utf-8")
	w.write("file_path,comment_content,comment_end_line_number,matched_label,related_code\n")
	#print(len(file_paths))
	#exit()
	file_cnt = 0
	for i in file_paths:
		file_cnt += 1
		if file_cnt % 10==0:
			print("finished %d"%file_cnt)
		s, lines_cnt = ecp.code_file_split(i)
		labels = ecp.simple_label(s)
		res = ecp.match_code(s, labels, lines_cnt)
		for comment,code,label,cnt in res:
			w.write(",".join([i, comment.replace('\n', "[enter]").replace(',', '[comma]'), str(cnt), label, code.replace('\n', "[enter]").replace(',', '[comma]')]) + '\n')
		
	w.close()