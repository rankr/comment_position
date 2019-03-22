#coding: utf-8

import time
import func
import extract_comment_position as ecp
import re


#global 
re_class_def = re.compile("^(\s)*class( )+(\w)+")
re_struct = re.compile("^struct ")
re_func_def = re.compile("^(\s)*(\w| |<|>|:|\*|\~)+(\w)+\(.*\)(\n|\{)")
re_func_decl = re.compile("^(\s)*(\w| |<|>|:|\*|\~)+(\w)+\(.*\);")
re_for = re.compile("^(\s)*for\(.*;.*;.*\)")
re_while = re.compile("^(\s)*while(\s)*\(.*\)") 
re_if = re.compile("^(\s)*if(\s)*\(.+\)") #checked
re_else = re.compile("^(\s)*else")
re_macro = re.compile("^\#")
re_call = re.compile("(\w)+\(.*\)")#this re should use re.search not re.match
re_empty = re.compile("^(\s|\{|\})*$")
re_var_def = re.compile("^(\s)*(\w)+((\w)|<|>|\:|(\*)|(\s))*( )+((\w)|<|>|\:|(\*)|(\s))*(\w)+(\[(\w)+\])?(\s)*(=|;)") #考虑了指针，模板，数组，namespace的情况




def simple_label(splited_list):#set a label for each snippet
    base_time = time.time()
    labels = []
    for i in splited_list:

        t = re.match(re_class_def, i)
        u = time.time()
        print(1)
        print(u-base_time)
        base_time = u

        re.match(re_struct, i)
        u = time.time()
        print(2)
        print(u-base_time)
        base_time = u

        re.match(re_func_def, i)
        u = time.time()
        print(3)
        print(u-base_time)
        base_time = u

        re.match(re_func_decl, i)
        u = time.time()
        print(4)
        print(u-base_time)
        base_time = u

        re.match(re_for, i)
        u = time.time()
        print(5)
        print(u-base_time)
        base_time = u

        re.match(re_while, i)
        u = time.time()
        print(6)
        print(u-base_time)
        base_time = u

        re.match(re_if, i)
        u = time.time()
        print(7)
        print(u-base_time)
        base_time = u

        re.match(re_else, i)
        u = time.time()
        print(8)
        print(u-base_time)
        base_time = u

        re.match(re_macro, i)
        u = time.time()
        print(9)
        print(u-base_time)
        base_time = u

        re.match(re_empty, i)
        u = time.time()
        print(10)
        print(u-base_time)
        base_time = u

        re.match(re_var_def, i)
        u = time.time()
        print(11)
        print(u-base_time)
        base_time = u

        re.search(re_call, i)
        u = time.time()
        print(12)
        print(u-base_time)
        base_time = u

    return labels



a = ["void CV_AccumBaseTest::get_test_array_types_and_sizes( int test_case_idx,\n                        vector<vector<Size> >& sizes, vector<vector<int> >& types )\n{\n"]

b = time.time()
r = simple_label(a)
print(time.time()-b)