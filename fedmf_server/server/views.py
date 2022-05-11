from django.http import HttpResponse
import numpy as np
from django.shortcuts import render
from server import parameter
# Create your views here.


#user_rating is a list of dict


# define model and initalize

def init(request):
    item_vector = np.random.normal(size=[parameter.num_items, parameter.hidden_dim])
    np.save('item_vector.npy', item_vector)
    np.save('gradient_dict.npy', {})
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
    for g in gradients:
        item_vector = item_vector - parameter.lr * (-2 * g + 2 * parameter.reg_u * item_vector)
    np.savetxt('item_vector.npy', item_vector)
    return HttpResponse("success")