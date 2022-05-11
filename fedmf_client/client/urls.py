from django.urls import path

from client import views



urlpatterns = [
    path('init', views.init, name='init'),
    path('data/preprocess', views.data_processing, name='preprocess data'),
    path('data/user_vector', views.load_user_vector, name='user_vector'),
    path('update', views.loacal_update, name='local update'),
    path('predict', views.predict, name='predict'),
]