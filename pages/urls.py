from django.urls import path
from .import  views

app_name = 'pages'

urlpatterns=[
     path('',views.home_page, name='home'),
     path('contact_us', views.contact_page, name = 'contact_us'),
     path('signup/<str:sub_type>', views.SubscriptionSignUp.as_view(), name = 'subscription_signup'),
     path('group_management', views.group_management, name='group_management'),
     path('edit_group_user/<int:group_id>/<int:id>', views.GroupUserEditView.as_view(), name = 'edit_group_user'),
     #path('edit_group_user/<int:id>', views.edit_user, name = 'edit_group_user'),
     path('search_items',views.search_items, name = 'search_items'),
]