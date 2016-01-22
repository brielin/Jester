import numpy as np
import sys

class sliding_window:
    def __init__(self, win_len=100, arr_len=1000, vec_len=1000, A=None):
        if A is not None:
            if A.shape[1] != vec_len:
                sys.stderr.write("Initializing array doesn't match params\n")
                sys.exit(1)
            else:
                self.A = np.vstack((A,np.zeros((arr_len-A.shape[0],vec_len))))
                self.a_pos = A.shape[0]
        else:
            self.A = np.zeros((arr_len,vec_len))
            self.a_pos = 0
        self.arr_len=arr_len
        self.vec_len=vec_len
        self.win_len=win_len

    def next(self, v):
        if len(v) != self.vec_len:
            sys.stderr.write('Given vector does not match vector size\n')
            sys.exit(1)
        self.A[self.a_pos]=v
        self.a_pos+=1
        if self.a_pos <= self.win_len:
            ret = self.A[0:self.a_pos]
        else:
            ret = self.A[(self.a_pos-self.win_len):self.a_pos]
        if self.a_pos==self.arr_len:
            self.__init__(self.win_len,self.arr_len,self.vec_len,
                          A=self.A[(self.a_pos-self.win_len):self.a_pos])
        ## This effects a copy, but it doesn't slow things down much
        return ret[::-1]
