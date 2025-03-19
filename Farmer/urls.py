from django.urls import path 
from .import views 

urlpatterns = [
    path('index',views.index,name='index'),
    path('about',views.about,name='about'),
    path('contact',views.contact,name='contact'),
    path('service',views.service,name='service'),
    path('register',views.register,name='reg'),
    path('login',views.login,name='login'),
    path('logout',views.logout,name='logout'),
    path('product',views.product,name='product'),
    path('crop',views.crop,name='cropprediction'),
    path('fertilizer',views.fertilizer,name='fertilizerprediction'),
    path('predict',views.predict,name='predict'),
]
