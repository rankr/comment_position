# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:13 2019

@author: Guo Yixuan
//modifier: Pengcheng Li
"""

import re

#统计代码块的复杂度，度量指标有代码行数、大括号最深层数、变量个数、运算符密集程度
def complexity_measure(string):
    #行数:line_num
    line_num = string.count('\n') + 1#some functions just have one line
    #print("line number is :{}".format(line_num))
    
    #大括号的最大层数:max_depth
    max_depth = 0
    count_depth = 0
    for each_char in string:
        if each_char is '{':
            count_depth += 1
        if each_char is '}':
            count_depth -= 1
        if count_depth > max_depth:
            max_depth = count_depth
    #print("max bracket depth is :{}".format(max_depth))
    
    #变量个数，检测在此模块定义的变量的个数，包括指针，
    #有个疑问是如何加入检测自定义的struct变量，这个有点困难
    int_pattern = re.compile("int")
    float_pattern = re.compile("float")
    double_pattern = re.compile("double")
    struct_pattern = re.compile("struct")
    
    #TODO: 改成所有非保留字的识别，和括号前识别认为是函数调用
    int_num = len(int_pattern.findall(string))
    float_num = len(float_pattern.findall(string))
    double_num = len(double_pattern.findall(string))
    struct_num = len(struct_pattern.findall(string))
    variable_total_num = int_num + float_num + double_num + struct_num
    #print("basic variable number is:", variable_total_num)
    
    #运算符密集度(算术运算，位运算，单位：个/行)
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
    #统计乘号和除号前现将代码段按行切片,先预处理用666替换注释里的'//'和'/*'和'*/'还有'///'
    code_line_array = re.split('\n',string)
    for each_line in code_line_array:
        if('/*' in each_line):
            each_line.replace("/*","666")
        if('*/' in each_line):
            each_line.replace("*/","666")
        if('//' in each_line):
            each_line.replace("//","666")
        if('///' in each_line):
            each_line.replace("///","666")
        if('||' in each_line):
            each_line.replace("||","666")
        if('&&' in each_line):
            each_line.replace("&&","666")
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
    #print("total operator number is:",operator_total_num)
    op_line_rate = operator_total_num/line_num
    return [line_num, max_depth, variable_total_num, operator_total_num, op_line_rate]

if __name__ == '__main__':
    #test code as:
    complexity_measure("void VariationalRefinementImpl::mergeCheckerboard(Mat &dst, RedBlackBuffer &src)\n{\nint buf_j, j;\nfor (int i = 0; i < dst.rows; i++)\n{\nfloat *src_r_buf = src.red.ptr<float>(i + 1);\nfloat *src_b_buf = src.black.ptr<float>(i + 1);\nfloat *dst_buf = dst.ptr<float>(i);\nbuf_j = 1;\n\n if (i % 2 == 0)\n{\nfor (j = 0; j < dst.cols - 1; j += 2)\n{\n dst_buf[j] = src_r_buf[buf_j];\n  dst_buf[j + 1] = src_b_buf[buf_j];\n      buf_j++;\n  }\nif (j < dst.cols)\ndst_buf[j] = src_r_buf[buf_j];\n}\nelse\n{\nfor (j = 0; j < dst.cols - 1; j += 2)\n{\ndst_buf[j] = src_b_buf[buf_j];\ndst_buf[j + 1] = src_r_buf[buf_j];\nbuf_j++;\n}\nif (j < dst.cols)\ndst_buf[j] = src_b_buf[buf_j];\n}\n}\n}\n")
