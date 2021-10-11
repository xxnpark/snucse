from rest_framework import serializers

from survey.models import OperatingSystem, SurveyResult

from user.serializers import UserSerializer

class SurveyResultSerializer(serializers.ModelSerializer):
    os = serializers.SerializerMethodField(read_only=True)
    user = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = SurveyResult
        fields = (
            'id',
            'os',
            'user',
            'python',
            'rdb',
            'programming',
            'major',
            'grade',
            'backend_reason',
            'waffle_reason',
            'say_something',
            'timestamp',
        )

    def get_os(self, survey):
        return OperatingSystemSerializer(survey.os, context=self.context).data

    def get_user(self, survey):
        if survey.user:
            return UserSerializer(survey.user, context=self.context).data
        else:  # user가 없는 경우 null 반환
            return None


class OperatingSystemSerializer(serializers.ModelSerializer):

    class Meta:
        model = OperatingSystem
        fields = (
            'id',
            'name',
            'description',
            'price',
        )