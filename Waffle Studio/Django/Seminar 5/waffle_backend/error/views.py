from rest_framework import permissions, status
from rest_framework.views import APIView, Response
from rest_framework.exceptions import APIException


class WrongAPIView(APIView):
    permission_classes = (permissions.AllowAny, )

    # GET /api/v1/wrong/
    def get(self, request):
        return Response(request)
#        raise APIException('500 error API example')
