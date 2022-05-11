from django.urls import path

from . import views


urlpatterns = [
    path('init', views.init, name='init'),
    path('gradient/upload', views.upload_gradient, name='user upload gradient'),
    path('gradient/all', views.get_gradient_list, name='get all gradients'),
    path('data/item_vector', views.download_item_vector, name='download item vector'),
    path('update', views.global_update, name='global update'),
]