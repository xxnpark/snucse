# Generated by Django 3.2.6 on 2021-10-06 18:29

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('seminar', '0002_auto_20210926_0418'),
    ]

    operations = [
        migrations.RenameField(
            model_name='seminar',
            old_name='updated_at_column',
            new_name='updated_at',
        ),
        migrations.RenameField(
            model_name='userseminar',
            old_name='updated_at_column',
            new_name='updated_at',
        ),
        migrations.AlterField(
            model_name='userseminar',
            name='seminar',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='user_seminars', to='seminar.seminar'),
        ),
        migrations.AlterField(
            model_name='userseminar',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='user_seminars', to=settings.AUTH_USER_MODEL),
        ),
    ]