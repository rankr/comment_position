# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:13 2019

@author: Guo Yixuan
"""

import re

#获取代码块行数，被complexity_measure(string)调用
def get_line_num(string):
    line_num = len(re.findall('\n',string))
    print("line number is :{}".format(line_num))
    return line_num

#获取代码块最大深度，进入一个大括号深度就加1，被complexity_measure(string)调用
def get_max_depth(string):
    max_depth = 0
    count_depth = 0
    for each_char in string:
        if each_char is '{':
            count_depth += 1
        if each_char is '}':
            count_depth -= 1
        if count_depth > max_depth:
            max_depth = count_depth
    print("max bracket depth is :{}".format(max_depth))
    return max_depth

#预处理函数将代码段按行切片,用666替换注释里的'//' '/*' '*/' '///' 'for' 'do' 'while'
#返回处理后的list
def preprocess(string): 
    lines = re.split('\n',string)
    for i in range(len(lines)):
        if('/*' in lines[i]):
            lines[i]=lines[i].replace("/*","666")
        if('*/' in lines[i]):
            lines[i]=lines[i].replace("*/","666")
        if('//' in lines[i]):
            lines[i]=lines[i].replace("//","666")
        if('///' in lines[i]):
            lines[i]=lines[i].replace("///","666")
        if('||' in lines[i]):
            lines[i]=lines[i].replace("||","666")
        if('&&' in lines[i]):
            lines[i]=lines[i].replace("&&","666")
    return lines

def get_variable_num(string):
    lines = preprocess(string)
    key_word_list = ['void',  'auto',  'short',  'long',   'int','float','double',  'char', 'struct',
                    'class',   'if' ,  'else' ,'switch',  'case',  'for',    'do', 'while', 'return'
                     'goto','continue','break', 'const', 'sizeof', 'return', 'typdef', 'extern','enum',
                     'register', 'default','union', 'static', 'volatile']
    variable_num = 0
    var_name_pattern = re.compile("\w+")
    
    variable_list = []
    for i in range(len(lines)):
        for each_key_word in key_word_list:
            #去掉行中出现的关键词
            if each_key_word in lines[i]:
                lines[i] = lines[i].replace(each_key_word, "")
        #把所有的变量名（包括函数名）放在同一列表variable_list
        variable_list += var_name_pattern.findall(lines[i])
    #去掉variable_list中重复元素
    no_repeat_list = list(set(variable_list))
    #print(no_repeat_list)

    #把数字开头的元素从no_repeat_list列表中去掉
    for i in range(len(no_repeat_list)):
        r = re.match('^\d',no_repeat_list[i])
        #r为none时，说明出现数字开头
        if r:
            no_repeat_list[i] = ""    
    #删除空字符串
    no_repeat_list.remove("")
    #print(no_repeat_list)
    variable_num = len(no_repeat_list)
    return variable_num

#获得运算符数目
def get_operator_num(string):
    code_line_array = preprocess(string)
    #运算符考虑(算术运算+-*/%，位运算|&)
    plus_pattern = re.compile("\+")
    plus_plus_pattern = re.compile("\+\+")
    product_pattern = re.compile("\*")
    division_pattern = re.compile("\/")
    modulo_pattern = re.compile("\%")
    bit_or_pattern = re.compile("\|")
    bit_and_pattern = re.compile("\&")
    #(1)运算符'+'号数量，去掉++的数量
    plus_num = len(plus_pattern.findall(string))
    plus_plus_num = len(plus_plus_pattern.findall(string))
    plus_num = plus_num - plus_plus_num*2
    #(2)运算符'-'号数量，去掉--的数量
    minus_num = len(re.findall('-',string))
    minus_minus_num = len(re.findall('--',string))
    minus_num = minus_num - minus_minus_num*2
    
    #因为乘除法运算一般伴随着赋值操作，所以检测和'='同一行出现的乘号和除号的数量，
    #但是这样做无法取出指针*带来的误差
    #(3)运算符'*'号数量，
    product_num = 0
    #(4)运算符'/'号数量
    division_num = 0
    #(5)运算符'%'号数量
    modulo_num = 0
    #(6)位运算符'|''&'的数量
    bit_or_num = 0
    bit_and_num = 0
    for each_line in code_line_array:
        if ('*' in each_line) and ('=' in each_line):
            product_num += len(product_pattern.findall(each_line))
        if ('/' in each_line) and ('=' in each_line):
            division_num += len(division_pattern.findall(each_line))
        if ('%' in each_line) and ('=' in each_line):
            modulo_num += len(modulo_pattern.findall(each_line))
        if ('|' in each_line) and ('=' in each_line):
            bit_or_num += len(bit_or_pattern.findall(each_line))
        if ('&' in each_line) and ('=' in each_line):
            bit_and_num += len(bit_and_pattern.findall(each_line))
    #运算符总数
    operator_total_num = plus_num + minus_num + product_num + division_num + modulo_num + bit_or_num + bit_and_num
#    print("+:",plus_num)
#    print("-:",minus_num)
#    print("*:",product_num)
#    print("/:",division_num)
#    print("%:",modulo_num)
#    print("|:",bit_or_num)
#    print("&:",bit_and_num)
    print("total operator number is:",operator_total_num)
    return operator_total_num

#统计代码块的复杂度，度量指标有代码行数、大括号最深层数、变量个数、运算符密集程度
def complexity_measure(string):
    #行数:line_num
    line_num = get_line_num(string)
    
    #大括号的最大层数:max_depth
    max_depth = get_max_depth(string)
    
    #变量个数，检测在此模块定义的变量的个数，包括指针，
    variable_total_num = get_variable_num(string)
    
    #运算符密集度(算术运算，位运算，单位：个/行)
    operator_total_num = get_operator_num(string)
    
    op_line_rate = operator_total_num/line_num
    print("operator density is:%.2f each line"%(op_line_rate))
    return [line_num, max_depth, variable_total_num, operator_total_num, op_line_rate]

if __name__ == '__main__':
    #test code as:
    complexity_measure("void VariationalRefinementImpl::mergeCheckerboard(Mat &dst, RedBlackBuffer &src)\n{\nint buf_j, j;\nfor (int i = 0; i < dst.rows; i++)\n{\nfloat *src_r_buf = src.red.ptr<float>(i + 1);\nfloat *src_b_buf = src.black.ptr<float>(i + 1);\nfloat *dst_buf = dst.ptr<float>(i);\nbuf_j = 1;\n\n if (i % 2 == 0)\n{\nfor (j = 0; j < dst.cols - 1; j += 2)\n{\n dst_buf[j] = src_r_buf[buf_j];\n  dst_buf[j + 1] = src_b_buf[buf_j];\n      buf_j++;\n  }\nif (j < dst.cols)\ndst_buf[j] = src_r_buf[buf_j];\n}\nelse\n{\nfor (j = 0; j < dst.cols - 1; j += 2)\n{\ndst_buf[j] = src_b_buf[buf_j];\ndst_buf[j + 1] = src_r_buf[buf_j];\nbuf_j++;\n}\nif (j < dst.cols)\ndst_buf[j] = src_b_buf[buf_j];\n}\n}\n}\n")
