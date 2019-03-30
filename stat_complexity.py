#coding: utf-8

import re
import os
import pandas as pd

import func
import extract_comment_position as ecp
import complexity_measure as cm

if __name__ == "__main__":
    c = pd.read_csv("./result/opencv_func_with_comment_stat.csv")
    n = pd.read_csv("./result/opencv_func_without_comment_stat.csv")
    
    c_op_dense = list(c['operator_total_num'])
    c_op_dense.sort()
    lc = len(c_op_dense)
    print("***Func operator_total_num with comment:\n\tmean: %f\n\tquarter: %f\n\tmedian: %f\n\t\
        3quarter: %f\n"%(sum(c_op_dense)/lc, c_op_dense[lc//4], c_op_dense[lc//2], c_op_dense[lc//4*3]))


    n_op_dense = list(n['operator_total_num'])
    n_op_dense.sort()
    ln = len(n_op_dense)
    print("***Func operator_total_num without comment:\n\tmean: %f\n\tquarter: %f\n\tmedian: %f\n\t\
        3quarter: %f\n"%(sum(n_op_dense)/ln, n_op_dense[ln//4], n_op_dense[ln//2], n_op_dense[ln//4*3]))

