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

import os
import pathlib
import json

from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.renderers import JSONRenderer
from rest_framework.views import APIView
from rest_framework.response import Response

from python_functions import data_reader, power_flow_solver

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


def tools_page(request):
    form = FileUpload()
    path = 'media/storage/group_{0}/user_{1}/'.format(request.user.group_id, request.user.id)
    if os.path.exists(path):
        user_files = os.listdir(path)
    else:
        os.makedirs(path)
        user_files = os.listdir(path)
    return render(request,'../templates/power_system_tools/tools_page.html',{'form':form,'user_data':request.user,'user_files':user_files})


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