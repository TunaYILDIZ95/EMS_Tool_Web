from curses.ascii import US
from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin

#from EMS_Database.models import User_Table
# Register your models here.
from .models import *


# class AccountInline(admin.StackedInline):
#     model = User_Table
#     can_delete = False
#     verbose_name_plural = 'User_Table'

# class CustomizedUserAdmin (UserAdmin):
#     inlines = (AccountInline, )


# admin.site.unregister(User)
# admin.site.register(User, CustomizedUserAdmin)

admin.site.register(MainFunctionTable)
admin.site.register(MainFunctionSubscriptionTable)
admin.site.register(SubFunctionTable)
admin.site.register(SubFunctionSubscriptionTable)
#admin.site.register(User_Table)