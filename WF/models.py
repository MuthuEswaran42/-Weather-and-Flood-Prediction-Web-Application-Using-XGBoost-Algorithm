from django.contrib.auth.models import User
from django.db import models

class ImageModel(models.Model):
    image = models.ImageField(upload_to='images/')
    link = models.URLField(blank=True, null=True)
    description = models.CharField(max_length=255)
    user = models.ForeignKey(User, on_delete=models.CASCADE) # Track uploader

    def __str__(self):
        return self.description
