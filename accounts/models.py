from django.db import models
from django.contrib.auth.models import AbstractUser
from django_countries.fields import CountryField

# Create your models here.


class User(AbstractUser):
    first_name = models.CharField(max_length=255)  # Field name made lowercase.
    last_name = models.CharField(max_length=255)  # Field name made lowercase.
    email = models.EmailField(max_length=255,unique=True)  # Field name made lowercase.
    country = CountryField() # Field name made lowercase.
    city = models.CharField(max_length=255)  # Field name made lowercase.
    site = models.CharField(max_length=100)  # Field name made lowercase.
    company = models.CharField(max_length=255)  # Field name made lowercase.
    sub_type = models.CharField(max_length=100, 
        choices=[('Silver','Silver'),('Gold','Gold'),('Platinum','Platinum'),('Guest','Guest')],default='Guest')  # Field name made lowercase.
    role = models.CharField(max_length=100, blank=True, null=True,
        choices=[('Guest','Guest'),('Manager','Manager'),('User','User'),('Admin','Admin')],
        default='Guest')  # Field name made lowercase.
    language_id = models.BigIntegerField()  # Field name made lowercase.
    group_id = models.BigIntegerField()  # Field name made lowercase.
    storage_size = models.BigIntegerField()
    total_user_number = models.BigIntegerField(blank=True, null=True)  # Field name made lowercase.
    remaining_user_number = models.BigIntegerField(blank=True, null=True)  # Field name made lowercase.
    sub_start_date = models.DateField(blank=True, null=True)  # Field name made lowercase.
    sub_end_date = models.DateField(blank=True, null=True)  # Field name made lowercase.
    payment_status = models.BooleanField(default=False)  # Field name made lowercase.
    owner_status = models.BooleanField(default=False)  # Field name made lowercase.
    change_by = models.CharField(max_length=255, blank=True, null=True)  # Field name made lowercase.
    change_date = models.DateField(blank=True, null=True)  # Field name made lowercase.

    REQUIRED_FIELDS = ['first_name', 'last_name','email', 'country','city', 'site', 'company', 'role','language_id', 'group_id',
        'total_user_number', 'remaining_user_number','storage_size',]

class Storage(models.Model):
    storage_size_id = models.IntegerField(primary_key = True)
    sub_type = models.CharField(max_length=100, 
        choices=[('Silver','Silver'),('Gold','Gold'),('Platinum','Platinum'),('Guest','Guest')],default='Guest')
    storage_limit = models.BigIntegerField()