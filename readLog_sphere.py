import string
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def readLog(filename):

    f = open(filename)
    l = np.zeros((1,2))
    flag_eval_line = False

    for line in f.readlines():
        
        if '**** EPOCH' in line:  ### which epoch??
            #line.strip()  
            #**** EPOCH 058 **** 
            epoch_cnt = line.strip().split(' ')[-2]   
            epoch_cnt = int(epoch_cnt)
            
            if epoch_cnt < 400:
                if epoch_cnt%50 == 0:
                    epoch = epoch_cnt################### every 10 epoch

            else:
                if epoch_cnt%20 == 0:
                    epoch = epoch_cnt##################  every 5 epoch

        if flag_eval_line:####    split the epe result
            epe = line.strip().split(' ')[-1]
            epe = float(epe)

            tmp = np.array([[epoch, epe]])#shape:(1,2) 

            l = np.append(l, tmp, axis=0)#(n,2)
            flag_eval_line = False

        if 'eval whole scene mean' in line:  ####  evalauting line is the next line
            flag_eval_line = True
        


    fname_txt = os.path.join(BASE_DIR, 'result_epe.txt')##### save the result as txt

    np.savetxt(fname_txt, l, fmt='%.06f')##save the result as txt

    print('done')

if __name__ == '__main__':
    # readLog('bao.txt')
    readLog('log_train.txt')
