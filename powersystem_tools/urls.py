from django.urls import path
from .import  views



urlpatterns=[
     path('powerflow_analysis/', views.powerflow_analysis, name = 'powerflow_analysis'),
     path('state_estimation/', views.state_estimation, name = 'state_estimation'),
     path('load_folder/', views.load_folder, name='load_folder'),
     path("run_state_estimation/", views.run_state_estimation, name="run_state_estimation"),
     path('api/graph/data/', views.GraphData.as_view()),
     path('api/pf_solution/data/', views.PowerFlowSolution.as_view()),
     path('upload_files/',views.upload_files, name = 'upload_files'),
]