from django.db import models
from accounts.models import User
from django.core.validators import FileExtensionValidator

# Create your models here.
def get_upload_path(instance,filename):
    return 'storage/group_{0}/user_{1}/{2}'.format(instance.user.group_id, instance.user.id, filename)

class Document(models.Model):
    user = models.ForeignKey(User, null=True, on_delete=models.CASCADE)
    docfile = models.FileField(upload_to=get_upload_path, validators=[FileExtensionValidator(allowed_extensions=["dat","csv","xlsx","txt"])])

    def delete(self):
        self.docfile.storage.delete(self.docfile.name)
        super().delete()