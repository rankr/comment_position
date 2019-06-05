#coding: utf-8

import time
import re
import os
import functools

import func
import extract_comment_position as ecp

def dir_cmt_stat():
    p1 = "/Users/apple/Documents/work1/cpp_repos/linux-master/security"
    
    pw = "./result/linux_dir_security.csv"
    
    w = open(pw, 'w')
    w.write("dir,cmt rate,code,comment\n")
    
    for i in os.listdir(p1):
        p = p1 + '/' + i
        if not os.path.isdir(p):
            continue
        arr= func.cpp_file_select(p)
        code = 0
        comment = 0
        for file in arr:
            a = ''
            with open(file) as f:
                a = f.read()
            b = func.code_line_num(a)
            code += b[0]
            comment += b[1]
        if code == 0:
            print("file "+p+"  zero!!")
            w.write(p+",NO CODE\n")
        else:
            print("Dir %s:\tcmt rate: %f\tcode: %d\tcomment: %d\t"%(p, comment/(code+comment), code, comment))
            w.write(','.join([p, str(comment/(code+comment)), str(code), str(comment)]) + '\n')

def dir_func_cmt_stat(prefix = "../cpp_repos/linux-master/drivers"):
    """
    result: avg func commented rate: 0.1897543125980136
    three subdir: block, ntb, hid
    we choose block to do study, cause it's largest among three
    """
    h = {}
    with open("./result/linux_func_with_comment.csv") as f1:
        a = f1.readlines()
        for i in a:
            b = i.split(',')
            if b[0].startswith(prefix):
                b[0] = b[0][len(prefix)+1:]
                subdir = b[0].split('/')[0]
                if os.path.isdir(prefix + '/' + subdir):
                    if subdir in h:
                        h[subdir][0] += 1
                    else:
                        h[subdir] = [1, 0]
    with open("./result/linux_func_without_comment.csv") as f2:
        a = f2.readlines()
        for i in a:
            b = i.split(',')
            if b[0].startswith(prefix):
                b[0] = b[0][len(prefix)+1:]
                subdir = b[0].split('/')[0]
                if os.path.isdir(prefix + '/' + subdir):
                    if subdir in h:
                        h[subdir][1] += 1
                    else:
                        h[subdir] = [0, 1]
    return h


def mycmp(a, b):
    if a[1] > b[1]:
        return -1
    elif a[1] < b[1]:
        return 1
    return 0;

def dir_avg_cplx_func():
    h = {}
    p = "./result/linux_func_with_comment.csv"
    head = '../cpp_repos/linux-master/'
    
    with open(p) as f:
        f.readline()
        a = f.readlines()
        a = [i.strip().split(',') for i in a]
        for i in a:
            path = i[0]
            foo = i[1]
            d = path[len(head):]
            d = d.split('/')[0]
            if d in h:
                h[d].append((foo, func.code_line_num(foo)[0], path))
            else:
                h[d] = [(foo, func.code_line_num(foo)[0], path)]
    
    w = open('./result/linux_avg_cplx_func_cmt.csv', 'w')
    w.write('dir,path,len,func\n')
    for i in h:
        if len(h[i]) != 0:
            h[i] = sorted(h[i], key=functools.cmp_to_key(mycmp))
            cnt = 0
            for j in h[i]:
                w.write(i + ',' + j[2][len(head):] + ',' + str(j[1]) + ',' + j[0] + '\n')
                cnt += 1
                if cnt == 5:
                    break
    w.close()




