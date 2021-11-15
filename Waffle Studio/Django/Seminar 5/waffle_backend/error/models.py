from django.contrib.auth import get_user_model
from django.db import models
from django.utils import timezone

User = get_user_model()


class ErrorLog(models.Model):
    user = models.ForeignKey(User, related_name='error_user', on_delete=models.CASCADE, null=True)  # 현재 로그인 유저
    time = models.DateTimeField(default=timezone.now)  # 에러가 발생학 시각
    status_code = models.IntegerField()  # 에러코드
    message = models.CharField(max_length=200)  # 에러 메시지 (내용)
    request_method = models.CharField(max_length=10, null=True)  # 에러가 발생한 요청의 method
    request_API = models.CharField(max_length=50, null=True)  # 에러가 발생한 API
