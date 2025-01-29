from urllib import request
from django.shortcuts import redirect, render, get_object_or_404
from django.views.generic.edit import FormView
from django.contrib.auth import get_user_model

from accounts.models import User
from .form import FileUpload
from django.urls import reverse_lazy
from .models import Document
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import subprocess

import os
import pathlib
import json

from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.renderers import JSONRenderer
from rest_framework.views import APIView
from rest_framework.response import Response

from python_functions import data_reader, power_flow_solver
from python_functions.StateEstimation.main_with_classes import StateEstimator

### Python Libraries
from dss import dss


def get_user_storage(user):
    user_files = os.listdir('media/storage/group_{0}/user_{1}/'.format(user.group_id, user.id))
    print(user_files)
    total = sum(os.path.getsize('media/storage/group_{0}/user_{1}/{2}'.format(user.group_id, user.id, file)) for file in user_files)/(1024*1024)
    print(total)
    print(user.storage_size)
    return total < user.storage_size

# def upload_files(request):
#     # get_user_storage(request.user)
#     print('askldgjaklsdgjklasjdgklasjkdlg')
#     print(request.FILES.get('docfile'))
#     if request.method == "POST" and request.FILES.get('docfile'):
#         form = FileUpload(request.POST, request.FILES)
#         files = request.FILES.getlist('docfile')
#         if form.is_valid():          
#             for f in files:
#                 file_instance = Document(docfile=f, user = request.user)
#                 file_instance.save()
#         return redirect('home')
#     else:
#         form = FileUpload()
#         print('olmadi')
#     return redirect('home')


def powerflow_analysis(request):
    form = FileUpload()
    path = 'media/storage/group_{0}/user_{1}/'.format(request.user.group_id, request.user.id)
    if os.path.exists(path):
        user_files = os.listdir(path)
    else:
        os.makedirs(path)
        user_files = os.listdir(path)
    return render(request,'../templates/power_system_tools/powerflow_analysis.html',{'form':form,'user_data':request.user,'user_files':user_files})


def state_estimation(request):
    user_storage_path = f'media/storage/group_{request.user.group_id}/user_{request.user.id}/'
    example_storage_path = 'media/examples/'

    # Get user files (files only)
    user_files = []
    if os.path.exists(user_storage_path):
        user_files = [
            {"name": f, "is_folder": False}  # User files are always files
            for f in os.listdir(user_storage_path)
            if os.path.isfile(os.path.join(user_storage_path, f))
        ]

    # Get example files & folders
    example_items = []
    if os.path.exists(example_storage_path):
        example_items = [
            {"name": f, "is_folder": os.path.isdir(os.path.join(example_storage_path, f))}
            for f in os.listdir(example_storage_path)
        ]

    # Check if the request comes from JavaScript (fetch)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({"user_files": user_files, "example_files": example_items})

    # Normal HTML page rendering
    context = {
        'user_files': user_files,
        'example_items': example_items,
        'example_base_path': example_storage_path,
        'user_data': request.user
    }
    return render(request, 'power_system_tools/state_estimation.html', context)


def upload_files(request):
    if request.method == "POST" and request.FILES.get('docfile'):
        form = FileUpload(request.POST, request.FILES)
        files = request.FILES.getlist('docfile')
        if form.is_valid():          
            for f in files:
                file_instance = Document(docfile=f, user = request.user)
                file_instance.save()
        user_files = os.listdir('media/storage/group_{0}/user_{1}/'.format(request.user.group_id, request.user.id))
        return redirect('tools_page')
    else:
        form = FileUpload()
        return render(request, '../templates/power_system_tools/tools_page.html', {'form':form, 'user_data':request.user ,'user_files':user_files})
    
def load_folder(request):
    base_path = "media/examples/"  # Base path for examples
    folder_name = request.GET.get("folder", "")
    if folder_name == "..":
        # Navigate one level up
        parent_folder = os.path.dirname(base_path.rstrip("/"))
        folder_path = parent_folder
    else:
        folder_path = os.path.join(base_path, folder_name)

    if not os.path.exists(folder_path):
        return JsonResponse({"error": "Folder not found"}, status=404)

    # Retrieve subfolders & files
    folder_items = [
        {"name": f, "is_folder": os.path.isdir(os.path.join(folder_path, f))}
        for f in os.listdir(folder_path)
    ]

    return JsonResponse({"items": folder_items, "parent_folder": base_path})

def run_state_estimation(request):
    print(request.method)
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            selected_file = data.get("filename", "")

            if not selected_file:
                return JsonResponse({"error": "No file selected"}, status=400)

            # Define base paths
            user_storage_path = "media/storage/group_{}/user_{}/".format(request.user.group_id, request.user.id)
            example_storage_path = "media/examples/"

            # Determine if file is from "Your Files" or "Example Files"
            if os.path.exists(os.path.join(user_storage_path, selected_file)):  # Check in user files
                file_path = os.path.join(user_storage_path, selected_file)
            elif os.path.exists(os.path.join(example_storage_path, selected_file)):  # Check in example files
                file_path = os.path.join(example_storage_path, selected_file)
            else:
                return JsonResponse({"error": "File not found"}, status=404)

            # Run the Python script with the selected file as input
            try:
                print('python is started')
                main = StateEstimator(dss, badDataNumber = 0, seedCounter=0, fileLocation=file_path)
                result = main.solve()
                print('python is finished')
                # Extract required data
                table_data = {
                    "bus_list": main.SE.nodeOrder,  
                    "voltages": main.SE.voltageState.tolist(),  
                    "angles": main.SE.thetaState.tolist()  
                }
                
                return JsonResponse({"message": "Success", "table_data": table_data})
            except Exception as e:
                print(f'Error in State Estimation: {e}')
                    
            if result.returncode == 0:
                return JsonResponse({"message": "Success", "output": result.stdout})
            else:
                return JsonResponse({"error": "Script failed", "output": result.stderr}, status=500)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)

class GraphData(APIView):
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    permission_classes = [IsAuthenticated]
    renderer_classes = [JSONRenderer]

    # def get(self, request, format=None):
    #     # graph = {"nodes":[{ "name": "deneme" },{ "name": "deneme1", "parent": 0 },
    #     #                      { "name": "deneme2", "parent": 1 },{ "name": "deneme3", "parent": 2 },{ "name": "deneme3", "parent": 2 },{ "name": "deneme4", "parent": 1 }]}
                    
    #     # data = {
    #     #     "nodes" : graph['nodes'],
    #     # }
    #     graph = {"nodes":[{ "id": "Bus 10", "group":1, "value":1.01},{ "id": "Bus 20", "group":2, "value": 0.98}], 
    #     "links": [{"source": "Bus 10","target": "Bus 20", "value":1}]}
    #     data = graph
    #     return Response(data)
    def get(self,request, format = None):
        file_name = request.GET.get('file_name') # file name buraya gelecek
        file_location = 'media/storage/group_{0}/user_{1}/{2}'.format(request.user.group_id, request.user.id, file_name)
        foo = data_reader.DataReader("", file_location)
        foo.data_parses()
        foo.Y_bus_creation()
        asd = power_flow_solver.PowerFlow(foo)
        asd.power_flow_jacobian()
        asd.connectivity_creator()
        graph = {"nodes":asd.nodes, "links":asd.links}
        return Response(graph)

class PowerFlowSolution(APIView):
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    permission_classes = [IsAuthenticated]
    renderer_classes = [JSONRenderer]

    def get(self,request, format = None):
        file_name = request.GET.get('file_name') # file name buraya gelecek
        file_location = 'media/storage/group_{0}/user_{1}/{2}'.format(request.user.group_id, request.user.id, file_name)
        foo = data_reader.DataReader("", file_location)
        foo.data_parses()
        foo.Y_bus_creation()
        asd = power_flow_solver.PowerFlow(foo)
        asd.power_flow_jacobian()
        asd.connectivity_creator()
        table_data = {"bus_list":asd.BusList, "voltages":asd.V, "angles":asd.theta}

        return Response(table_data)