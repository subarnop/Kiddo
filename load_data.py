import numpy as np

class Load_data:
    def load_data():
        apple = np.load('data/apple.npy')    
        apple_x_train = apple[ : (apple.shape[0]*8//10) ]
        apple_x_test  = apple[apple.shape[0]*8//10 : ]
        apple_y_train = np.full((apple_x_train.shape[0], 1), 0)
        apple_y_test  = np.full((apple_x_test.shape[0], 1), 0)
        
        airplane = np.load('data/airplane.npy')
        airplane_x_train = airplane[ : airplane.shape[0]*8//10]
        airplane_x_test  = airplane[airplane.shape[0]*8//10 : ]
        airplane_y_train = np.full((airplane_x_train.shape[0], 1), 1)
        airplane_y_test = np.full((airplane_x_test.shape[0], 1), 1)
        
        beach = np.load('data/beach.npy')
        beach_x_train = beach[ : beach.shape[0]*8//10]
        beach_x_test  = beach[beach.shape[0]*8//10 : ]
        beach_y_train = np.full((beach_x_train.shape[0], 1), 1)
        beach_y_test = np.full((beach_x_test.shape[0], 1), 1)
    
        beard = np.load('data/beard.npy')
        beard_x_train = beard[ : beard.shape[0]*8//10]
        beard_x_test  = beard[beard.shape[0]*8//10 : ]
        beard_y_train = np.full((beard_x_train.shape[0], 1), 1)
        beard_y_test = np.full((beard_x_test.shape[0], 1), 1)
    
        car = np.load('data/car.npy')
        car_x_train = car[ : car.shape[0]*8//10]
        car_x_test  = car[car.shape[0]*8//10 : ]
        car_y_train = np.full((car_x_train.shape[0], 1), 1)
        car_y_test = np.full((car_x_test.shape[0], 1), 1)
    
        x_train = np.concatenate([apple_x_train, airplane_x_train, beach_x_train, beard_x_train, car_x_train])
        y_train = np.concatenate([apple_y_train, airplane_y_train, beach_y_train, beard_y_train, car_y_train])
        x_test  = np.concatenate([apple_x_test, airplane_x_test, beach_x_test, beard_x_test, car_x_test])
        y_test  = np.concatenate([apple_y_test, airplane_y_test, beach_y_test, beard_y_test, car_y_test])
    
        return x_train, y_train, x_test, y_test
    
