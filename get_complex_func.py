#coding: utf-8
import re

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



f_have = open("./result/linux_func_with_comment.csv")
f_not_have = open("./result/linux_func_without_comment.csv")

w1 = open("./result/linux_complex_sample_with_comment.csv", 'w')
w2 = open("./result/linux_complex_sample_without_comment.csv", 'w')
w1.write('path,line_num,func,comment\n')
w2.write('path,line_num,func\n')


a = f_have.readline()
while 1:
    a = f_have.readline()
    if not a:
        break
    path, func, comment = a.strip().split(',')
    c = code_line_num(func)
    if c>=500:
        w1.write(",".join([path, str(c), func, comment]) + '\n')

w1.close()

a = f_not_have.readline()
while 1:
    a = f_not_have.readline()
    if not a:
        break
    path, func = a.strip().split(',')
    c = code_line_num(func)
    if c>=500:
        w2.write(",".join([path, str(c), func]) + '\n')

w2.close()
