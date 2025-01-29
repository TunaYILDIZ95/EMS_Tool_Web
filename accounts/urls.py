from unicodedata import name
from django.urls import path
from .import  views

urlpatterns=[
     path('user_register/', views.user_register.as_view(), name= 'user_register'),
     path('delete_user/', views.delete_user,name='delete_user'),
     path('edit_profile/', views.UserEditView.as_view(), name = 'edit_user'),
     path('password/', views.PasswordChangeView.as_view(template_name='accounts/change_password.html'), name = 'password_change'),
     path('password_success', views.password_success, name = 'password_success'),
     #path('signup/', views.SignUp.as_view(), name= 'signup'),
     path('ajax/load_cities',views.load_cities, name= 'load_cities'),
     path('login', views.login_view, name= 'login'),
     path('logout', views.logout_view, name= 'logout'),
]