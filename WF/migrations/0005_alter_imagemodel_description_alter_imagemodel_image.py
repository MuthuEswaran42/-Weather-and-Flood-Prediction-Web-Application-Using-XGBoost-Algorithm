# Generated by Django 5.2 on 2025-04-22 02:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('WF', '0004_imagemodel_delete_uploadedimage'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagemodel',
            name='description',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='imagemodel',
            name='image',
            field=models.ImageField(upload_to='images/'),
        ),
    ]
