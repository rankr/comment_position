#coding: utf-8
import json
import pandas as pd
import re

import func as fc
#获取C语言关键词表
def get_key_word_list():
    key_word_list = ['void',  'auto',  'short',  'long',   'int','float','double',  'char', 'struct',
                    'class',   'if' ,  'else' ,'switch',  'case',  'for',    'do', 'while', 'return'
                     'goto','continue','break', 'const', 'sizeof', 'return', 'typdef', 'extern','enum',
                     'register', 'default','union', 'static', 'volatile']
    return key_word_list
kws = get_key_word_list()

#获得函数调用数量（去除重复）
def get_function_call(string):
    #普通形式调用，形式如：function(parameter 1);
    normal_call_pattern = re.compile("\w+\(")
    nomal_call_list = normal_call_pattern.findall(string)[1:]
    #prnt("normal:",nomal_call_list)
    
    #函数模板调用，形如：max<double>(2.0, 3.0);
    templet_call_pattern = re.compile("<\w+>\(")
    templet_call_list = templet_call_pattern.findall(string)
    #rint("templet:",templet_call_list)
    function_call_list = list(set(templet_call_list + nomal_call_list))
    function_call_list = [x for x in function_call_list if x not in kws]
    return function_call_list
def get_func_name(string):
    normal_call_pattern = re.compile("\w+\(")
    normal_call_list = normal_call_pattern.findall(string)
    if len(normal_call_list) == 0:
        print("SkipWarnning from get_func_name, string not detect func: ", string[0:100])
        return ''
    return normal_call_list[0]

func_cnt = json.load(open("/Users/apple/Documents/work1/code/result/linux_func_files.json"))

new_func_cnt = {}

c = pd.read_csv("/Users/apple/Documents/work1/code/result/linux_func_with_comment.csv")
for i in range(0, len(c)):
    item = c.loc[i]
    func = get_func_name(item['func'])
    if not func:
        continue
    if fc.code_line_num(item['func'])[0] > 20:
        new_func_cnt[func] = func_cnt[func]
    if i % 1000==0:
        print(i)


n = pd.read_csv("/Users/apple/Documents/work1/code/result/linux_func_without_comment.csv")
for i in range(0, len(n)):
    item = n.loc[i]
    func = get_func_name(item['func'])
    if not func:
        continue
    if fc.code_line_num(item['func'])[0] > 20:
        new_func_cnt[func] = func_cnt[func]
    if i % 1000==0:
        print(i)

json.dump(new_func_cnt, open("/Users/apple/Documents/work1/code/result/linux_func_files_filtered.json", 'w'))