# Generated by Django 3.2.6 on 2021-10-06 18:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0004_remove_instructorprofile_charge'),
    ]

    operations = [
        migrations.RenameField(
            model_name='instructorprofile',
            old_name='updated_at_column',
            new_name='updated_at',
        ),
        migrations.RenameField(
            model_name='participantprofile',
            old_name='updated_at_column',
            new_name='updated_at',
        ),
    ]