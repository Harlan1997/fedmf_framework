import numpy as np

# return error
def test(user_vector, item_vector, test_data):
    p = predict(user_vector, item_vector)
    real_label = []
    prediction = []
    real_label = [e[1] for e in test_data]
    prediction = [p[e[0]] for e in test_data]
    return real_label - prediction

def predict(request, user_vector, item_vector):
    return np.dot(user_vector, np.transpose(item_vector))