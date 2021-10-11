from django.urls import path, include
from rest_framework.routers import SimpleRouter
from .views import SeminarViewSet

router = SimpleRouter()
router.register('seminar', SeminarViewSet, basename='user')  # /api/v1/seminar/

urlpatterns = [
    path('', include(router.urls), name='auth-user')
]
