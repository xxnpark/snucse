from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.response import Response

from rest_framework.decorators import action

from survey.serializers import OperatingSystemSerializer, SurveyResultSerializer
from survey.models import OperatingSystem, SurveyResult

from datetime import datetime
from django.utils import timezone


class SurveyResultViewSet(viewsets.GenericViewSet):
    queryset = SurveyResult.objects.all()
    serializer_class = SurveyResultSerializer

    # POST /api/v1/survey/ : 새로운 유저 설문 결과 추가
    def create(self, request):
        try:
            python = int(request.data.get('python'))
            rdb = int(request.data.get('rdb'))
            programming = int(request.data.get('programming'))
            major = request.data.get('major')
            grade = request.data.get('grade')
            backend_reason = request.data.get('backend_reason')
            waffle_reason = request.data.get('waffle_reason')
            say_something = request.data.get('say_something')
            user = request.user
            os = request.data.get('os')

            # python, rgb, programming의 값이 1과 5 사이의 정수가 아닐 경우 400 반환
            if not (1<=python<=5 and 1<=rdb<=5 and 1<=programming<=5):
                return Response(status=status.HTTP_400_BAD_REQUEST)

            # 로그인되어있지 않은 경우 403 반환
            if not user.is_authenticated:
                return Response(status=status.HTTP_403_FORBIDDEN)

            operating_system, created = OperatingSystem.objects.get_or_create(name=os)
            survey = SurveyResult.objects.create(os=operating_system, user=user, python=python, rdb=rdb, programming=programming, major=major, grade=grade, backend_reason=backend_reason, waffle_reason=waffle_reason, say_something=say_something)
            return Response(self.get_serializer(survey).data, status=status.HTTP_201_CREATED)

        # python, rgb, programming 중 정수가 아닌 값이 있거나 비어 있을 경우 400 반환
        except ValueError:
            return Response(status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['GET'])
    def serialize(self, request):
        survey = SurveyResult.objects.get(id=int(request.data.get('id')))
        return Response(self.get_serializer(survey).data, status=status.HTTP_201_CREATED)

    def list(self, request):
        surveys = self.get_queryset().select_related('os')
        return Response(self.get_serializer(surveys, many=True).data)

    def retrieve(self, request, pk=None):
        survey = get_object_or_404(SurveyResult, pk=pk)
        return Response(self.get_serializer(survey).data)


class OperatingSystemViewSet(viewsets.GenericViewSet):
    queryset = OperatingSystem.objects.all()
    serializer_class = OperatingSystemSerializer

    def list(self, request):
        return Response(self.get_serializer(self.get_queryset(), many=True).data)

    def retrieve(self, request, pk=None):
        try:
            os = OperatingSystem.objects.get(id=pk)
        except OperatingSystem.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        return Response(self.get_serializer(os).data)
