import re
from urllib import request
from django.contrib.auth import login, logout,authenticate
from django.shortcuts import redirect, render
from django.contrib import messages
from django.views.generic import CreateView, UpdateView, DeleteView, DetailView
from .form import UserCreation, LoginForm, EditUserForm
from django.contrib.auth.forms import AuthenticationForm, UserChangeForm, PasswordChangeForm
from .models import User
from django.http import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils.decorators import method_decorator
from django.urls import reverse_lazy
from django.contrib.auth.views import PasswordChangeView

import json
from django.core import serializers

# Create your views here.

def notManagerUser(view_func):
    def wrapper_func(request, *args, **kwargs):
        if (request.user.role == 'Admin' or request.user.role == 'Manager') and request.user.remaining_user_number > 0:
            return view_func(request,*args,**kwargs)
        else:
            return redirect('home')        
    return wrapper_func

@method_decorator(notManagerUser, name='dispatch')
class user_register(CreateView):
    model = User
    form_class = UserCreation
    template_name = '../templates/accounts/user_register.html'

    def get_initial(self):
        return { 'country': self.request.user.country, 'site':self.request.user.site, 'company':self.request.user.company }

    def get_context_data(self, **kwargs):
        context = super(user_register, self).get_context_data(**kwargs)
        user_data = {'user_data':self.request.user}
        context.update(user_data)
        return context

    def form_valid(self, form):        
        user = form.save(group_id = self.request.user.group_id,language_id = self.request.user.language_id, sub_type = self.request.user.sub_type)
        #login(self.request, user)
        current_user = User.objects.get(pk = self.request.user.id)
        current_user.remaining_user_number = self.request.user.remaining_user_number-1
        current_user.save()
        return redirect('/pages/group_management') # giris yapildiginde yonlendirilecek ekranin uzantisi girilecek

# class SignUp(CreateView):
#     model = User
#     form_class = UserSignUp
#     template_name = '../templates/accounts/signup.html'

#     def form_valid(self, form):
#         user = form.save()
#         login(self.request, user)
#         return redirect('home') # giris yapildiginde yonlendirilecek ekranin uzantisi girilecek

class UserEditView(LoginRequiredMixin,UpdateView):
    form_class = EditUserForm
    template_name = '../templates/accounts/edit_profile.html'
    success_url: reverse_lazy('home')

    def get_object(self):    
        return self.request.user

    def get_context_data(self, **kwargs):
        context = super(UserEditView, self).get_context_data(**kwargs)
        user_data = {'user_data':self.request.user}
        context.update(user_data)
        return context
    
    def form_valid(self, form):
        user = form.save()
        messages.success(self.request,"Your Profile is Updated !")
        return(redirect('home'))

class PasswordChangeView(PasswordChangeView):
    form_class = PasswordChangeForm
    success_url = reverse_lazy('password_success')

    def get_context_data(self, **kwargs):
        context = super(PasswordChangeView, self).get_context_data(**kwargs)
        user_data = {'user_data':self.request.user}
        context.update(user_data)
        return context

def password_success(request):
    messages.success(request, "Your Password is Changed!" )
    return redirect('home')

def login_view(request):
    form = LoginForm(request.POST or None)
    if form.is_valid():
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        user = authenticate(username = username,password = password)
        login(request, user)
        user_data = request.user

        return redirect('home')
        #return render(request,'../templates/home.html',{'user_data':user_data}) # giris yapildiginde yonlendirilecek ekranin uzantisi girilecek
    return render(request,'accounts/form.html',{'form':form,'title':'Log In'})

def logout_view(request):
    logout(request)
    return redirect('home') # cikis yapildiginde gidilecek ekran buraya girilecek

def load_cities(request):
    json_data_cities = open('static/json_files/cities.json')
    city_data = json.load(json_data_cities) # deserialises it
    json_data_cities.close()
    country_name = request.GET.get('country_id')
    try:
        cities = city_data[country_name]
    except:
        cities = []
    change_status = request.GET.get('change_status')
    if request.user.is_authenticated and request.user.country == request.GET.get('country_name') and change_status == '0':
        user_data = request.user.city
        return JsonResponse({'cities':cities,'user_data':user_data}, safe=False)
    else:
        return JsonResponse({'cities':cities}, safe=False)

def delete_user(request):
    if request.method == 'POST':
        check_box_values = request.POST.getlist('checks[]')
        for values in check_box_values:
            user = User.objects.get(pk = int(values))
            user.delete()
            current_user = User.objects.get(pk = request.user.id)
            current_user.remaining_user_number += 1
            current_user.save()       
    return redirect('/pages/group_management')