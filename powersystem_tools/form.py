from turtle import color
from django import forms
from django.db import transaction
from .models import Document
from django.forms import ClearableFileInput

class FileUpload(forms.ModelForm):
    class Meta:
        model = Document
        fields = ['docfile']
        widgets = {
            'docfile': ClearableFileInput(attrs={'allow_multiple_selected': True}),
        }
    
    def __init__(self, *args, **kwargs):
        super(FileUpload, self).__init__(*args, **kwargs)
        self.fields['docfile'].label = "Files"