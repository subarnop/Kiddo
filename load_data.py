import numpy as np

#only for python 3.5 or above
import glob

class Load_data:
    def getNumberofClass():
       n=0
       for filename in glob.iglob('data/*.npy', recursive=True):
        n += 1
       
       print(n)
       return n
    def load_data():
        
        n = 0 
        for filename in glob.iglob('data/*.npy', recursive=True):
            
            data = np.load(filename)
            data_x_train = data[ : (data.shape[0]*8//10) ]
            data_x_test  = data[data.shape[0]*8//10 : ]
            data_y_train = np.full((data_x_train.shape[0], 1), n)
            data_y_test  = np.full((data_x_test.shape[0], 1), n)
            n = n+1
            x_train = np.concatenate([data_x_train, data_x_train])
            y_train = np.concatenate([data_y_train, data_y_train])
            x_test  = np.concatenate([data_x_test, data_x_test])
            y_test  = np.concatenate([data_y_test, data_y_test])
            
        return x_train, y_train, x_test, y_test
        
