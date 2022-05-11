from django.http import HttpResponse
from django.shortcuts import render
from yaml import load
import numpy as np
import json
import csv
import urllib.request as url_req
import json
from client import user_training, parameter

# Create your views here.

def init(request):
    user_vector = np.random.normal(size=[1, parameter.hidden_dim])
    np.save('user_vector.npy', user_vector)
    return HttpResponse('success to init user vector')

def load_user_vector(request):
    return HttpResponse(np.loadtxt('user_vector.npy'))

def data_processing(request, path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        ratings = [r for r in reader]
    user_id = ratings[0][0]
    rating_list = []
    for record in ratings:
        rating_list.append([int(record[1]), float(record[2]), int(record[3])])
    # sort by timestamp
    sorted_rating = sorted(rating_list, key=lambda x:x[-1], reverse=False)
    test_data = {user_id : sorted_rating[-3:]}
    return HttpResponse(rating_list)

def loacal_update(request, user_vector, rating_list, item_vector):
    user_vector, gradient = user_training.user_update(user_vector, rating_list, item_vector)
    np.savetxt('client/user_vector.txt', user_vector)
    return HttpResponse(gradient)

def predict(request, user_vector, item_vector):
    return HttpResponse(np.dot(user_vector, np.transpose(item_vector)))

    