#coding: utf-8

import os
import queue
import re

def cpp_file_select(directory):
    cpp_affix = ['cpp', 'h', 'hpp', 'cc', 'cxx', 'c']
    a = os.listdir(directory)
    q = queue.Queue()
    ret = []
    for i in a:
        q.put(directory+'/'+i)
    while not q.empty():
        u = q.get()
        if os.path.isdir(u):
            for i in os.listdir(u):
                q.put(u+'/'+i)
        else:
            affix = u.split('.')[-1]
            if affix.lower() in cpp_affix:
                ret.append(u)
    return ret

cmt1 = re.compile("\s*/\*")
cmt2 = re.compile("\s*//")
#only return the amount of the code, comment. but not content
def code_line_num(string):

    string = string.replace("[enter]", '\n')
    string = string.split('\n')
    code = 0
    comment = 0
    in_comment = 0
    for i in string:
        if not in_comment:
            if re.match(cmt1, i):
                comment += 1
                if '*/' not in i:
                    in_comment = 1
            elif '/*' in i:
                code += 1
                comment += 1
                if "*/" not in i:
                    in_comment += 1
            elif re.match(cmt2, i):
                comment += 1
                continue
            elif i.strip()!='':
                code += 1
        else:
            if '*/' in i:
                comment += 1
                in_comment = 0
    return code,comment

def simple_struct(string):
    string = string.replace("[enter]", '\n')
    total = string.count()