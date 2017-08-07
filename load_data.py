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
        x_train = np.empty([1, 784])
        y_train = np.empty([1, 1])
        x_test = np.empty([1, 784])
        y_test = np.empty([1, 1])
        
        for filename in glob.iglob('data/*.npy', recursive=True):
            
            data = np.load(filename)
            data_x_train = data[ : (data.shape[0]*8//10) ]
            data_x_test  = data[data.shape[0]*8//10 : ]
            data_y_train = np.full((data_x_train.shape[0], 1), n)
            data_y_test  = np.full((data_x_test.shape[0], 1), n)
            print(filename,n)
            n = n+1
            print(x_train.shape)
            print(data_x_train.shape)
            x_train = np.concatenate((x_train, data_x_train), axis=0)
            y_train = np.concatenate((y_train, data_y_train), axis=0)
            x_test  = np.concatenate((x_test, data_x_test), axis=0)
            y_test  = np.concatenate((y_test, data_y_test), axis=0)
            
        return x_train, y_train, x_test, y_test
        
