#coding: utf-8
import extract_comment_position as ecp

def extract_func(path):
    snippets, lines_cnt = ecp.code_file_split(path)
    labels = ecp.simple_label(snippets)
    matched = ecp.match_code(snippets, labels, lines_cnt)
    matched_posi = [x[4] for x in matched]
    ret = []
    for i, label, s in zip(range(0, len(snippets)), labels, snippets):
        if label == "func_def":
            posi = i
            braces = s.count('{') - s.count('}')
            while braces>0:
                posi += 1
                if posi == len(snippets):
                    print("From extract_func, Error occurred. Paht: "+ path +" is skipped")
                    return []
                temp_s = snippets[posi]
                braces += temp_s.count('{') - temp_s.count('}')
                s += temp_s
            flag = ''
            comment = ''
            if i in matched_posi:
                flag = 'Cmt'
                comment = matched[matched_posi.index(i)][0]
            else:
                flag = 'Not'
            ret.append((flag, s, comment))
    return ret