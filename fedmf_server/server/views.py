import imp
from django.http import HttpResponse
import numpy as np
from django.shortcuts import render
# Create your views here.


#user_rating is a list of dict

# define model and initalize

# todo log and virtualization and test

def config(request, num_items, num_users, hidden_dim, lr, reg_u, reg_v):
    config_dict = {'num_items' : num_items, 'num_users' : num_users, 'hidden_dim' : hidden_dim, 
                        'lr' : lr, 'reg_u' : reg_u, 'reg_v' : reg_v}
    np.save('config.npy', config_dict)
    return HttpResponse("success to add config")

def init(request):
    config_dict = np.load('config.npy', allow_pickle=True).item()
    item_vector = np.random.normal(size=[config_dict['num_items'], config_dict['num_users']])
    np.save('item_vector.npy', item_vector)
    np.save('gradient_dict.npy', {})
    np.save('error_dict', {})
    return HttpResponse("success to init item vector and gradient dict")


def upload_gradient(request, gradient, user_id):
    new_added = {user_id : gradient}
    gradient_dict = np.load('gradient_dict.npy', allow_pickle=True).item()
    gradient_dict.update(new_added)
    np.save('gradient_dict.npy', gradient_dict)
    return HttpResponse("success to upload gradient")

def get_gradient_list(request):
    gradient_dict = np.load('gradient_dict.npy', allow_pickle=True).item()
    return HttpResponse(list(gradient_dict.values()))

def download_item_vector(request):
    return HttpResponse(np.loadtxt('item_vector.npy'))

def global_update(request, gradients, item_vector):
    config_dict = np.load('config.npy', allow_pickle=True).item()
    lr = config_dict['lr']
    reg_u = config_dict['reg_u']
    for g in gradients:
        item_vector = item_vector - lr * (-2 * g + 2 * reg_u * item_vector)
    np.save('item_vector.npy', item_vector)
    return HttpResponse("success")

def clear_gradient(request):
    empty_dict = {}
    np.save('gradient_dict.npy', empty_dict)

def gradient_full(request):
    config_dict = np.load('config.npy', allow_pickle=True).item()
    num_users = config_dict['num_users']
    gradient_dict = np.load('gradient_dict.npy').item()
    if len(gradient_dict.values()) == num_users:
        return HttpResponse('collect enough gradients')
    else:
        return HttpResponse('gradient not enough')

def upload_error(request, error, user_id):
    new_added = {user_id : error}
    error_dict = np.load('gradient_dict.npy', allow_pickle=True).item()
    error_dict.update(new_added)
    np.save('gradient_dict.npy', error_dict)
    return HttpResponse("success to upload error")