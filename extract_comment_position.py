#coding: utf-8
import re
import os
import json
import queue
import time

fk = 0
def code_file_split(path):#返回按照{,},\n,//,/*分割后的文件，类型：list
    global fk
    a = ''
    with open(path) as f:
        a = f.readlines()

    ret = []
    lines_cnt = []
    now_code = ''
    now_comment = ''
    in_block_comment = 0
    in_line_comment = 0

    i = 1
    if not a:
        return [],[]
    line = a[0].lstrip()
    while True:
        '''
        just judge for unlimited loop
        fk += 1
        if fk == 100000:
            break
        '''
        if line == '':
            if i == len(a):
                if now_comment:
                    ret.append(now_comment)
                    lines_cnt.append(i)
                break
            line = a[i].lstrip(' ').lstrip('\t')
            i = i+1
        if not in_block_comment and not in_line_comment:
            if '/*' in line:
                in_block_comment = 1
                posi = line.find('/*')
                if posi != 0:#before the block comment, there is no code
                    now_code += line[:posi]
                    ret.append(now_code)
                    lines_cnt.append(i)
                    now_code = ''
                line = line[posi:]
            elif '//' in line:
                posi = line.find('//')
                if posi != 0:
                    now_code += line[:posi]
                    ret.append(now_code)
                    lines_cnt.append(i)
                    now_code = ''
                in_line_comment = 1
                now_comment = line[posi:]
                line = ''
            else:#just code
                if line=='\n':
                    if now_code=='':
                        ret.append('\n')
                        lines_cnt.append(i)
                    else:
                        ret.append(now_code)
                        lines_cnt.append(i)
                        ret.append('\n')
                        lines_cnt.append(i)
                    now_code = ''
                    line = ''
                elif '{' in line or '}' in line:
                    #print("in Kuohao line is:  "+line)
                    p1 = line.find('{')
                    p2 = line.find('}')
                    if p2==-1 or (p1!=-1 and p1<p2):
                        posi = p1
                        if line[posi+1] == '\n':
                            now_code += line[:posi+2]
                            ret.append(now_code)
                            lines_cnt.append(i)
                            now_code = ''
                            line = line[posi+2:]
                        else:
                            now_code += line[:posi+1]
                            ret.append(now_code)
                            lines_cnt.append(i)
                            now_code = ''
                            line = line[posi+1:]
                            #print("from after {, line is:'%s'"%line)
                    else:
                        posi = p2
                        if len(line)==posi+1 or line[posi+1] == '\n':
                            now_code += line[:posi+2]
                            ret.append(now_code)
                            lines_cnt.append(i)
                            now_code = ''
                            line = line[posi+2:]
                        else:
                            now_code += line[:posi+1]
                            ret.append(now_code)
                            lines_cnt.append(i)
                            now_code = ''
                            line = line[posi+1:]
                else:
                    now_code += line
                    line = ''
        elif in_block_comment:
            if '*/' in line:
                in_block_comment = 0
                posi = line.find('*/')
                if len(line)==posi+2 or line[posi+2]=='\n':
                    now_comment += line
                    ret.append(now_comment)
                    lines_cnt.append(i)
                    now_comment = ''
                    line = ''
                else:
                    now_comment += line[:posi+2]
                    ret.append(now_comment)
                    lines_cnt.append(i)
                    now_comment = ''
                    line = line[posi+2:]
            else:
                now_comment += line
                line = ''
        else:#in line comment
            posi = line.find('//')
            if posi == -1:
                ret.append(now_comment)
                lines_cnt.append(i)
                in_line_comment = 0
                now_comment = ''
                continue
            if posi != 0:
                ret.append(now_comment)
                lines_cnt.append(i)
                now_comment = ''
                now_code = line[:posi]
                ret.append(now_code)
                lines_cnt.append(i)
                now_code = ''
                now_comment = line[posi:]
            else:
                now_comment += line
            line = ''
    return ret, lines_cnt


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
re_var_def = re.compile("^(\s)*(\w)+((\w)|<|>|\:|(\*)|(\s))*( )+((\w)|<|>|\:|(\*)|(\s))*(\w)+(\[(\w)+\])?(\s)*(=|;|,)") #考虑了指针，模板，数组，namespace的情况



def simple_label(splited_list):#set a label for each snippet
    labels = []
    for i in splited_list:
        if '/*' in i or '//' in i:
            labels.append('comment')
        elif re.match(re_class_def, i):
            labels.append('class')
        elif re.match(re_struct, i):
            labels.append('struct')
        elif re.match(re_func_def, i):
            labels.append('func_def')
        elif re.match(re_func_decl, i):
            labels.append('func_decl')
        elif re.match(re_for, i):
            labels.append('for')
        elif re.match(re_while, i):
            labels.append('while')
        elif re.match(re_if, i):
            labels.append('if')
        elif re.match(re_else, i):
            labels.append('else')
        elif re.match(re_macro, i):
            labels.append('macro')
        elif re.match(re_empty, i):
            labels.append('empty')
        elif re.match(re_var_def, i):
            labels.append('var_def')
        elif re.search(re_call, i):
            labels.append('call')
        else:
            labels.append('other')
    return labels



def match_code(splited_list, labels, lines_cnt):#match each comment to a code snippet
    total = len(splited_list)

    matched = []

    for i in range(0, total):
        label = labels[i]
        snippet = splited_list[i]
        matched_label = ''
        matched_snippet = ''
        if label == 'comment':
            if i==0:
                matched_label = 'HEAD'
            elif i==total-1:
                matched_label = labels[i-1]
                matched_snippet = splited_list[i-1]
            else:
                if splited_list[i-1][-1] != '\n':
                    matched_label = labels[i-1]
                    matched_snippet = splited_list[i-1]
                elif not re.match(re_empty, splited_list[i+1]):
                    matched_label = labels[i+1]
                    matched_snippet = splited_list[i+1]
                else:#没有匹配到任何代码 —— 规则待完善，先看看有多少
                    matched_label = 'NOTHING'
                    matched_snippet = ''
            matched.append((snippet, matched_snippet, matched_label, lines_cnt[i]))
    return matched


