#coding: utf-8

import os
import queue

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
