#coding: utf-8
import os

import func_match as fm
import func

#突然发现我的文件结构和你们不一样，我的result、opencv都在其他目录，你们注意改改


if __name__ == "__main__":
    path = "../cpp_repos/linux-master"
    file_paths = func.cpp_file_select(path)


    if not os.path.exists('result'):
        os.mkdir('result')

    w_have = open("./result/linux_func_with_comment.csv", 'w')
    w_not_have = open("./result/linux_func_without_comment.csv", 'w')

    w_have.write("path,func,comment\n")
    w_not_have.write("path,func\n")

    file_cnt = 0
    for i in file_paths:
        file_cnt += 1
        if file_cnt % 10==0:
            print("finished %d"%file_cnt)
        res = fm.extract_func(i)
        for flag, code, comment in res:
            if flag == 'Cmt':
                w_have.write(",".join([i, code.replace('\n', "[enter]").replace(',', '[comma]'), comment.replace('\n', "[enter]").replace(',', '[comma]')]) + '\n')
            elif flag == 'Not':
                w_not_have.write(",".join([i, code.replace('\n', "[enter]").replace(',', '[comma]')]) + '\n')
            else:
                assert(0)
    w_have.close()
    w_not_have.close()
