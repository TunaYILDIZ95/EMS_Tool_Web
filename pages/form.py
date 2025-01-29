import re
from typing_extensions import Required
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django import forms
from accounts.models import User
from django.db import transaction
from django.contrib.auth import authenticate

import json
import datetime



class GroupEditUserForm(UserChangeForm):
    username = forms.CharField(max_length=100)
    first_name = forms.CharField(max_length=255)
    last_name = forms.CharField(max_length=255)
    email = forms.EmailField(max_length=255)  # Field name made lowercase.

    json_data_country = open('static/json_files/countries.json')
    country_data = json.load(json_data_country) # deserialises it
    country_tuple = [tuple([item['code'],item['name']]) for item in country_data]
    json_data_country.close()

    country = forms.ChoiceField(choices = country_tuple,initial='AF') # Field name made lowercase.
    city = forms.CharField(widget=forms.Select(choices=[]))
    site = forms.CharField(max_length=100)  # Field name made lowercase.
    company = forms.CharField(max_length=255)  # Field name made lowercase.

    sub_type = forms.CharField(max_length=255)
    role = forms.CharField(max_length=255)

    storage_size = forms.CharField(max_length=255)

    class Meta:
        model = User
        fields = ['username', 'password', 'first_name', 'last_name', 'email', 'country', 'city', 'site', 'company', 'sub_type', 'role', 'storage_size']

    def __init__(self, *args, **kwargs):
       super(GroupEditUserForm, self).__init__(*args, **kwargs)
       self.fields['sub_type'].widget.attrs['readonly'] = True
       self.fields['role'].widget.attrs['readonly'] = True
       self.fields['storage_size'].widget.attrs['readonly'] = True

class SubscriptionSignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=255)
    last_name = forms.CharField(max_length=255)
    email = forms.EmailField(max_length=255)  # Field name made lowercase.

    json_data_country = open('static/json_files/countries.json')
    country_data = json.load(json_data_country) # deserialises it
    country_tuple = [tuple([item['code'],item['name']]) for item in country_data]
    json_data_country.close()

    country = forms.ChoiceField(choices = country_tuple,initial='AF') # Field name made lowercase.
    city = forms.CharField(widget=forms.Select(choices=[]))
    site = forms.CharField(max_length=100)  # Field name made lowercase.
    company = forms.CharField(max_length=255)  # Field name made lowercase.
    sub_type = forms.CharField(max_length=100)
    #sub_type = forms.ChoiceField(choices=[('Guest','Guest'),('Silver','Silver'),('Gold','Gold'),('Platinum','Platinum')],initial='Guest')  
    #role = forms.ChoiceField(choices=[('Guest','Guest'),('Manager','Manager'),('User','User'),('Admin','Admin')],initial='User')  # Field name made lowercase.

    class Meta(UserCreationForm.Meta):
        model = User

    @transaction.atomic
    def save(self,**kwargs):
        group_id_last = max(User.objects.values_list('group_id'))[0]
        user = super().save(commit=False)
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.email = self.cleaned_data.get('email')
        user.country = self.cleaned_data.get('country')
        user.city = self.cleaned_data.get('city')
        user.site = self.cleaned_data.get('site')
        user.company = self.cleaned_data.get('company')
        user.sub_type = kwargs['sub_type']
        user.role = kwargs['role']
        user.language_id = 1
        user.group_id = int(group_id_last) + 1
        user.storage_size = kwargs['storage_size']
        user.payment_status = False
        user.owner_status = False
        user.total_user_number = kwargs['total_user_number']
        user.remaining_user_number = kwargs['remaining_user_number']
        if type(kwargs['sub_start_date']) is datetime.date:
            user.sub_start_date = kwargs['sub_start_date']
            user.sub_end_date = kwargs['sub_end_date']
        user.save()
        return user
    
    def __init__(self, *args, **kwargs):
       super(SubscriptionSignUpForm, self).__init__(*args, **kwargs)
       self.fields['sub_type'].widget.attrs['readonly'] = True