from rest_framework.views import exception_handler, Response
from rest_framework.exceptions import ValidationError, APIException, ErrorDetail
from django.http import HttpResponse
from error.models import ErrorLog


def custom_exception_handler(exc, context):
    request = context['request']

    data = {
        'Detail': exc.detail,
        'Request method': request.method,
        'Request API': request.META.get('PATH_INFO')
    }

    ErrorLog.objects.create(
        user=request.user if not request.user.is_anonymous else None,
        status_code=exc.status_code,
        message=exc.detail,
        request_method=request.method,
        request_API=request.META.get('PATH_INFO')
    )
    return Response(data, status=exc.status_code, headers=[])


def server_error(request):
    ErrorLog.objects.create(
        user=request.user if not request.user.is_anonymous else None,
        status_code=500,
        message="500 Server Error",
        request_method=request.method,
        request_API=request.META.get('PATH_INFO')
    )
    return HttpResponse('A server error occurred. Please contact the administrator.', status=500)
