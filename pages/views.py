from django.shortcuts import redirect, render, get_object_or_404
from django.contrib import messages
from django.views.generic import CreateView, UpdateView, DeleteView, DetailView
from accounts.models import User, Storage
from django.http import HttpResponseNotFound
from django.contrib.auth.mixins import LoginRequiredMixin
from .form import GroupEditUserForm, SubscriptionSignUpForm
from django.urls import reverse_lazy
from django.contrib.auth import login, logout,authenticate
from django.utils.decorators import method_decorator

from datetime import date
from dateutil.relativedelta import relativedelta
from itertools import chain

# Create Decorator
def notManagerUser(view_func):
    def wrapper_func(request, *args, **kwargs):
        if (request.user.role == 'Admin' or request.user.role == 'Manager') and kwargs.get("group_id") == request.user.group_id:
            return view_func(request,*args,**kwargs)
        else:
            messages.warning(request, "Your Role is not satisfied for this page !!!!")
            return redirect('home')        
    return wrapper_func

def notRoleType(view_func):
    def wrapper_func(request, *args, **kwargs):
        if (request.user.role == 'Admin' or request.user.role == 'Manager'):
            return view_func(request,*args,**kwargs)
        else:
            messages.warning(request, "Your Role is not satisfied for this page !!!!")
            return redirect('home')        
    return wrapper_func

# Create your views here.
def home_page(request):
    if request.user.is_anonymous:
        user_data = {'sub_type':'Guest'}
    else:
        user_data = request.user
    return render(request, '../templates/home.html', {'user_data':user_data})

def contact_page(request):
    return render(request, '../templates/pages/contactus.html',{'user_data':request.user})

class SubscriptionSignUp(CreateView):
    model = User
    form_class = SubscriptionSignUpForm
    template_name = '../templates/pages/subscription_signup.html'

    def get_initial(self, **kwargs):      
        return {'sub_type': self.kwargs.get("sub_type")}

    def form_valid(self, form):
        if self.kwargs.get("sub_type") == 'Guest':
            total_user_number = 0
            remaining_user_number = 0
            role = 'Guest'
            sub_start_date = ''
            sub_end_date = ''
            storage_size = Storage.objects.get(sub_type = self.kwargs.get("sub_type")).storage_limit
        elif self.kwargs.get("sub_type") == 'Silver':
            total_user_number = 5
            remaining_user_number = 5
            role = 'Manager'
            sub_start_date = date.today()
            sub_end_date = date.today()+relativedelta(months = 1)
            storage_size = Storage.objects.get(sub_type = self.kwargs.get("sub_type")).storage_limit
        elif self.kwargs.get("sub_type") == 'Gold':
            total_user_number = 10
            remaining_user_number = 10
            role = 'Manager'
            sub_start_date = date.today()
            sub_end_date = date.today()+relativedelta(months = 1)
            storage_size = Storage.objects.get(sub_type = self.kwargs.get("sub_type")).storage_limit
        elif self.kwargs.get("sub_type") == 'Platinum':
            total_user_number = 30
            remaining_user_number = 30
            role = 'Manager'
            sub_start_date = date.today()
            sub_end_date = date.today()+relativedelta(months = 1)
            storage_size = Storage.objects.get(sub_type = self.kwargs.get("sub_type")).storage_limit
        user = form.save(sub_type = self.kwargs.get("sub_type"),storage_size = storage_size , total_user_number = total_user_number, remaining_user_number = remaining_user_number,
         role = role, sub_start_date=sub_start_date, sub_end_date = sub_end_date)
        login(self.request, user)
        return redirect('home') # giris yapildiginde yonlendirilecek ekranin uzantisi girilecek

def group_management(request):      
    if request.user.is_authenticated and (request.user.role == "Admin" or request.user.role == "Manager"):         
        user_data = request.user
        group_data1 = User.objects.filter(group_id = request.user.group_id, role = 'User')
        group_data2 = User.objects.filter(group_id = request.user.group_id, role = 'Guest')
        group_data = sorted(chain(group_data1, group_data2),key=lambda instance: instance.username)

        data = {"user_data":user_data,"group_data":group_data}
        #user_data = User.objects.filter(pk = request.user.id)
        #print(user_data)
        return render(request, '../templates/pages/group_management.html',{'data':data})
    else:  
        messages.warning(request,'You need to login first or requirments are not satisfied !') # rengine bakilacak
        return redirect('home')

@method_decorator(notManagerUser, name='dispatch')
class GroupUserEditView(LoginRequiredMixin,UpdateView):
    form_class = GroupEditUserForm
    template_name = '../templates/pages/edit_group_user.html'
    success_url: reverse_lazy('home')

    def get_object(self, **kwargs): 
        selected_user_object = get_object_or_404(User, pk=self.kwargs.get("id"))
        return selected_user_object

    def get_context_data(self, **kwargs):
        context = super(GroupUserEditView, self).get_context_data(**kwargs)
        user_data = {'user_data':self.request.user}
        context.update(user_data)
        return context
    
    def form_valid(self, form):
        user = form.save()
        messages.success(self.request,"User Profile is Updated !")
        return(redirect('home'))


def search_items(request):
    if request.user.is_authenticated and (request.user.role == "Admin" or request.user.role == "Manager"):
        user_data = request.user
        search_data = request.GET.get('qsearch')
        
        if len(search_data) != 0:
            if len(User.objects.filter(username__contains = search_data, group_id = request.user.group_id)) != 0:
                group_data1 = User.objects.filter(username__contains = search_data, role = 'User', group_id = request.user.group_id)
                group_data = sorted(chain(group_data1),key=lambda instance: instance.username)
            elif len(User.objects.filter(email__contains = search_data, group_id = request.user.group_id)) != 0:
                group_data1 = User.objects.filter(email__contains = search_data, role = 'User', group_id = request.user.group_id)
                group_data = sorted(chain(group_data1),key=lambda instance: instance.username)
            elif len(User.objects.filter(first_name__contains = search_data, group_id = request.user.group_id)) != 0:
                group_data1 = User.objects.filter(first_name__contains = search_data, role = 'User', group_id = request.user.group_id)
                group_data = sorted(chain(group_data1),key=lambda instance: instance.username)
            elif len(User.objects.filter(last_name__contains = search_data, group_id = request.user.group_id)) != 0:
                group_data1 = User.objects.filter(last_name__contains = search_data, role = 'User', group_id = request.user.group_id)
                group_data = sorted(chain(group_data1),key=lambda instance: instance.username)
        else:
            group_data1 = User.objects.filter(group_id = request.user.group_id, role = 'User')
            group_data2 = User.objects.filter(group_id = request.user.group_id, role = 'Guest')
            group_data = sorted(chain(group_data1, group_data2),key=lambda instance: instance.username)


        data = {"user_data":user_data,"group_data":group_data}
        return render(request, '../templates/pages/group_management.html',{'data':data})
    else:
        return render(request, '../templates/pages/group_management.html',{})