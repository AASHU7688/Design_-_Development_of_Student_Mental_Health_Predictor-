from django.urls import path
from home import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='home'),
    path('about',views.about,name='about'),
    path('contact',views.contact,name='about'),
    path('prediction',views.prediction,name='prediction'),
    path('login',views.login,name='login'),
    path('signup',views.signup,name='signup'),
    path('chatbot',views.chatbot,name='chatbot'),

    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    # Define the profile view as well
    path('profile/', views.profile, name='profile')

]
