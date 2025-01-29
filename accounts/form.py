import re
from typing_extensions import Required
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django import forms
from .models import User, Storage
from django.db import transaction
from django.contrib.auth import authenticate

import json

# Make custom choice field for no validation requested with ajax in front end
# class ChoiceFieldNoValidation(forms.ChoiceField):
#     def validate(self, value):
#         pass


class UserCreation(UserCreationForm):
    first_name = forms.CharField(max_length=255)
    last_name = forms.CharField(max_length=255)
    email = forms.EmailField(max_length=255)  # Field name made lowercase.

    json_data_country = open('static/json_files/countries.json')
    country_data = json.load(json_data_country) # deserialises it
    country_tuple = [tuple([item['code'],item['name']]) for item in country_data]
    json_data_country.close()

    country = forms.ChoiceField(choices = country_tuple,initial='AF') # Field name made lowercase.
    #city = ChoiceFieldNoValidation()
    city = forms.CharField(widget=forms.Select(choices=[]))  # Field name made lowercase.
    site = forms.CharField(max_length=100)  # Field name made lowercase.
    company = forms.CharField(max_length=255)  # Field name made lowercase.

    class Meta(UserCreationForm.Meta):
        model = User

    @transaction.atomic
    def save(self,**kwargs):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.email = self.cleaned_data.get('email')
        user.country = self.cleaned_data.get('country')
        user.city = self.cleaned_data.get('city')
        user.site = self.cleaned_data.get('site')
        user.company = self.cleaned_data.get('company')
        user.sub_type = kwargs['sub_type']
        user.role = 'User'
        user.language_id = kwargs['language_id']
        user.group_id = kwargs['group_id']
        storage_size_instance = Storage.objects.get(sub_type = kwargs['sub_type'])
        user.storage_size = storage_size_instance.storage_limit
        user.payment_status = False
        user.owner_status = False
        user.save()
        return user

# class UserSignUp(UserCreationForm):
#     first_name = forms.CharField(max_length=255)
#     last_name = forms.CharField(max_length=255)
#     email = forms.EmailField(max_length=255)  # Field name made lowercase.

#     json_data_country = open('static/json_files/countries.json')
#     country_data = json.load(json_data_country) # deserialises it
#     country_tuple = [tuple([item['code'],item['name']]) for item in country_data]
#     json_data_country.close()

#     country = forms.ChoiceField(choices = country_tuple,initial='AF') # Field name made lowercase.
#     #city = ChoiceFieldNoValidation()  # custom field with no validation.
#     city = forms.CharField(widget=forms.Select(choices=[]))
#     site = forms.CharField(max_length=100)  # Field name made lowercase.
#     company = forms.CharField(max_length=255)  # Field name made lowercase.
#     #sub_type = forms.ChoiceField(choices=[('Guest','Guest'),('Silver','Silver'),('Gold','Gold'),('Platinum','Platinum')],initial='Guest')  
#     #role = forms.ChoiceField(choices=[('Guest','Guest'),('Manager','Manager'),('User','User'),('Admin','Admin')],initial='User')  # Field name made lowercase.

#     class Meta(UserCreationForm.Meta):
#         model = User

#     @transaction.atomic
#     def save(self):
#         group_id_last = max(User.objects.values_list('group_id'))[0]
#         user = super().save(commit=False)
#         user.first_name = self.cleaned_data.get('first_name')
#         user.last_name = self.cleaned_data.get('last_name')
#         user.email = self.cleaned_data.get('email')
#         user.country = self.cleaned_data.get('country')
#         user.city = self.cleaned_data.get('city')
#         user.site = self.cleaned_data.get('site')
#         user.company = self.cleaned_data.get('company')
#         user.sub_type = 'Guest'
#         user.role = 'Guest'
#         user.language_id = 1
#         user.group_id = int(group_id_last) + 1
#         user.payment_status = False
#         user.owner_status = False
#         user.total_user_number = 5
#         user.remaining_user_number = 5
#         user.save()
#         return user

class LoginForm(forms.Form):
    username = forms.CharField(max_length=100,label='Username')
    password = forms.CharField(max_length=100,label='Password',widget=forms.PasswordInput)

    @transaction.atomic
    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')
        if username and password:
            user = authenticate(username=username, password=password)
            if not user:
                raise forms.ValidationError("Username or Password is incorrect!")
        return super(LoginForm, self).clean()


class EditUserForm(UserChangeForm):
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

    total_user_number = forms.CharField(max_length=255)
    remaining_user_number = forms.CharField(max_length=255)

    sub_start_date = forms.DateField()  # Field name made lowercase.
    sub_end_date = forms.DateField()  # Field name made lowercase.

    class Meta:
        model = User
        fields = ['username', 'password', 'first_name', 'last_name', 'email', 'country', 'city', 'site', 'company', 'sub_type', 'role','total_user_number','remaining_user_number',
        'storage_size','sub_start_date','sub_end_date']

    def __init__(self, *args, **kwargs):
       super(EditUserForm, self).__init__(*args, **kwargs)
       self.fields['sub_type'].widget.attrs['readonly'] = True
       self.fields['role'].widget.attrs['readonly'] = True
       self.fields['total_user_number'].widget.attrs['readonly'] = True
       self.fields['storage_size'].widget.attrs['readonly'] = True
       self.fields['remaining_user_number'].widget.attrs['readonly'] = True
       self.fields['sub_start_date'].widget.attrs['readonly'] = True
       self.fields['sub_end_date'].widget.attrs['readonly'] = True