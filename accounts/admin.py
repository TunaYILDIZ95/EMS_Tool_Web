from django.contrib import admin
from .models import User, Storage
# Register your models here.
from django.contrib.auth.admin import UserAdmin


class UserAdminConfig(UserAdmin):

    ordering = ('id',)
    list_display = ('username','email','first_name','last_name')

    fieldsets = (
        (None, {'fields': ('username', 'email', 'first_name','last_name','password')}),
        ('Permissions', {'fields': ('is_staff', 'is_active','is_superuser')}),
        ('Personal', {'fields': ('country','city','site','company','sub_type','role','language_id','group_id','storage_size','total_user_number',
            'remaining_user_number','sub_start_date','sub_end_date','payment_status','owner_status','change_by','change_date',)}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'first_name', 'last_name', 'password1', 'password2', 'is_active', 'is_staff',
                'country','city','site','company','sub_type','role','language_id','group_id','storage_size','total_user_number',
                'remaining_user_number','sub_start_date','sub_end_date','payment_status','owner_status','change_by','change_date',)}
         ),
    )

admin.site.register(User,UserAdminConfig)
admin.site.register(Storage)