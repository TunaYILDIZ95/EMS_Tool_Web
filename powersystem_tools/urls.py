from django.urls import path
from .import  views



urlpatterns=[
     path('tools_page/', views.tools_page, name = 'tools_page'),
     path('api/graph/data/', views.GraphData.as_view()),
     path('api/pf_solution/data/', views.PowerFlowSolution.as_view()),
     path('upload_files/',views.upload_files, name = 'upload_files'),
]