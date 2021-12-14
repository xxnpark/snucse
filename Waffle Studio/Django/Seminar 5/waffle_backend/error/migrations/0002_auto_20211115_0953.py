# Generated by Django 3.2.6 on 2021-11-15 09:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('error', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='errorlog',
            name='request',
        ),
        migrations.AddField(
            model_name='errorlog',
            name='request_API',
            field=models.CharField(max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='errorlog',
            name='request_method',
            field=models.CharField(max_length=10, null=True),
        ),
    ]