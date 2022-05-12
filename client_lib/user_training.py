import numpy as np
import sys
import datetime
import os


def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
        
    fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d1')
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

def user_update(single_user_vector, user_rating_list, item_vector, config_dict):
    gradient = np.zeros([len(item_vector), len(single_user_vector)])
    lr = config_dict['lr']
    reg_u = config_dict['reg_u']
    for item_id, rate, _ in user_rating_list:
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        single_user_vector = single_user_vector - lr * (-2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = error * single_user_vector
    return single_user_vector, gradient

def loacal_update(user_vector, rating_list, item_vector, config_dict):
    user_vector, gradient = user_update(user_vector, rating_list, item_vector, config_dict)
    np.savetxt('user_vector.npy', user_vector)
    return gradient