import numpy as np

hidden_dim = 100

max_iteration = 1000

reg_u = 1e-4
reg_v = 1e-4

lr = 1e-4


def user_update(single_user_vector, user_rating_list, item_vector):
    gradient = np.zeros([len(item_vector), len(single_user_vector)])
    for item_id, rate, _ in user_rating_list:
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        single_user_vector = single_user_vector - lr * (-2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = error * single_user_vector
    return single_user_vector, gradient