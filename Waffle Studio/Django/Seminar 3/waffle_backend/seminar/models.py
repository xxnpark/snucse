from django.contrib.auth import get_user_model
from django.db import models
from django.utils import timezone
from datetime import time

User = get_user_model()

class Seminar(models.Model):
    name = models.CharField(max_length=20)
    capacity = models.IntegerField(default=0)
    count = models.IntegerField(default=0)
    time = models.TimeField(default=time(0, 0))
    online = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now_add=True)

class UserSeminar(models.Model):
    ROLE_CHOICES = (
        ('participant', 'participant'),
        ('instructor', 'instructor'),
    )

    user = models.ForeignKey(User, related_name='user_seminars', on_delete=models.CASCADE)
    seminar = models.ForeignKey(Seminar, related_name='user_seminars', on_delete=models.CASCADE)
    role = models.CharField(max_length=15, choices=ROLE_CHOICES, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    dropped_at = models.DateTimeField(null=True)
    updated_at = models.DateTimeField(auto_now=True)