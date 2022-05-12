import csv
import numpy as np

def init(hidden_dim):
    user_vector = np.random.normal(size=[1, hidden_dim])
    np.save('user_vector.npy', user_vector)

def load_user_vector():
    return np.load('user_vector.npy')

def data_processing(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        ratings = [r for r in reader]
    user_id = ratings[0][0]
    rating_list = []
    # [item, rating, timestamp]
    for record in ratings:
        rating_list.append([int(record[1]), float(record[2]), int(record[3])])
    # sort by timestamp
    sorted_rating = sorted(rating_list, key=lambda x:x[-1], reverse=False)
    test_data = sorted_rating[-3:]
    return rating_list, user_id, test_data