from django.urls import path
from .views import WrongAPIView

urlpatterns = [
    path('wrong/', WrongAPIView.as_view(), name='wrong'),  # /api/v1/wrong/
]
