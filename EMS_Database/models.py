from statistics import mode
from django.db import models
from django.contrib.auth.models import User
from django_countries.fields import CountryField



class MainFunctionSubscriptionTable(models.Model):
    function_name = models.CharField(db_column='Function_Name', max_length=255)  # Field name made lowercase.
    sub_type = models.CharField(db_column='Sub_Type', max_length=100)  # Field name made lowercase.
    status = models.BooleanField(db_column='Status')  # Field name made lowercase.
    function = models.ForeignKey('MainFunctionTable', models.DO_NOTHING, db_column='Function_ID')  # Field name made lowercase.

    class Meta:
        db_table = 'Main_Function_Subscription_Table'


class MainFunctionTable(models.Model):
    function_id = models.BigIntegerField(db_column='Function_ID', primary_key=True)  # Field name made lowercase.
    function_name = models.CharField(db_column='Function_Name', max_length=255)  # Field name made lowercase.

    class Meta:
        db_table = 'Main_Function_Table'


class SubFunctionSubscriptionTable(models.Model):
    sub_function = models.ForeignKey('SubFunctionTable', models.DO_NOTHING, db_column='Sub_Function_ID')  # Field name made lowercase.
    sub_function_name = models.CharField(db_column='Sub_Function_Name', max_length=255)  # Field name made lowercase.
    sub_type = models.CharField(db_column='Sub_Type', max_length=100)  # Field name made lowercase.
    status = models.BooleanField(db_column='Status')  # Field name made lowercase.

    class Meta:
        db_table = 'Sub_Function_Subscription_Table'


class SubFunctionTable(models.Model):
    function = models.ForeignKey(MainFunctionTable, models.DO_NOTHING, db_column='Function_ID')  # Field name made lowercase.
    function_name = models.CharField(db_column='Function_Name', max_length=255)  # Field name made lowercase.
    sub_function_id = models.BigIntegerField(db_column='Sub_Function_ID', primary_key=True)  # Field name made lowercase.
    sub_function_name = models.CharField(db_column='Sub_Function_Name', max_length=255)  # Field name made lowercase.

    class Meta:
        db_table = 'Sub_Function_Table'


# class User_Table(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     first_name = models.CharField(max_length=255)  # Field name made lowercase.
#     last_name = models.CharField(max_length=255)  # Field name made lowercase.
#     email = models.EmailField(max_length=255)  # Field name made lowercase.
#     country = CountryField() # Field name made lowercase.
#     city = models.CharField(max_length=255)  # Field name made lowercase.
#     site = models.CharField(max_length=100)  # Field name made lowercase.
#     company = models.CharField(max_length=255)  # Field name made lowercase.
#     sub_type = models.CharField(max_length=100, 
#         choices=[('Silver','Silver'),('Gold','Gold'),('Platinum','Platinum'),('Guest','Guest')],default='Guest')  # Field name made lowercase.
#     role = models.CharField(max_length=100, blank=True, null=True,
#         choices=[('Guest','Guest'),('Manager','Manager'),('User','User'),('Admin','Admin')],
#         default='Guest')  # Field name made lowercase.
#     language_id = models.BigIntegerField()  # Field name made lowercase.
#     group_id = models.BigIntegerField()  # Field name made lowercase.
#     total_user_number = models.BigIntegerField()  # Field name made lowercase.
#     remaining_user_number = models.BigIntegerField()  # Field name made lowercase.
#     sub_start_date = models.DateField(blank=True, null=True)  # Field name made lowercase.
#     sub_end_date = models.DateField(blank=True, null=True)  # Field name made lowercase.
#     payment_status = models.BooleanField()  # Field name made lowercase.
#     owner_status = models.BooleanField()  # Field name made lowercase.
#     change_by = models.CharField(max_length=255, blank=True, null=True)  # Field name made lowercase.
#     change_date = models.DateField(blank=True, null=True)  # Field name made lowercase.

#     def __str__(self) -> str:
#         return self.user.username
