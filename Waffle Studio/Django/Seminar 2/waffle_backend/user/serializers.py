from abc import ABC
from django.contrib.auth import get_user_model, authenticate
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import update_last_login
from rest_framework import serializers
from rest_framework_jwt.settings import api_settings
from user.models import ParticipantProfile, InstructorProfile
from seminar.models import UserSeminar
from seminar.serializers import ParticipantSeminarSerializer, InstructorSeminarSerializer

# 토큰 사용을 위한 기본 세팅
User = get_user_model()
JWT_PAYLOAD_HANDLER = api_settings.JWT_PAYLOAD_HANDLER
JWT_ENCODE_HANDLER = api_settings.JWT_ENCODE_HANDLER


# [ user -> jwt_token ] function
def jwt_token_of(user):
    payload = JWT_PAYLOAD_HANDLER(user)
    jwt_token = JWT_ENCODE_HANDLER(payload)
    return jwt_token


class UserCreateSerializer(serializers.Serializer):

    email = serializers.EmailField(required=True)
    username = serializers.CharField(required=True)
    password = serializers.CharField(required=True)
    first_name = serializers.CharField()
    last_name = serializers.CharField()

    def validate(self, data):
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        if bool(first_name) ^ bool(last_name):
            raise serializers.ValidationError("성과 이름 중에 하나만 입력할 수 없습니다.")
        if first_name and last_name and not (first_name.isalpha() and last_name.isalpha()):
            raise serializers.ValidationError("이름에 숫자가 들어갈 수 없습니다.")
        return data

    def create(self, validated_data):
        email = validated_data.get('email')
        username = validated_data.get('username')
        password = validated_data.get('password')
        first_name = validated_data.get('first_name')
        last_name = validated_data.get('last_name')
        user = User.objects.create_user(email=email, username=username, password=password, first_name=first_name, last_name=last_name)
        return user, jwt_token_of(user)


class UserLoginSerializer(serializers.Serializer):

    email = serializers.CharField(max_length=64, required=True)
    password = serializers.CharField(max_length=128, write_only=True)
    token = serializers.CharField(max_length=255, read_only=True)

    def validate(self, data):
        email = data.get('email', None)
        password = data.get('password', None)
        user = authenticate(email=email, password=password)

        if user is None:
            raise serializers.ValidationError("이메일 또는 비밀번호가 잘못되었습니다.")

        update_last_login(None, user)
        return {
            'email': user.email,
            'token': jwt_token_of(user)
        }

class UserSerializer(serializers.ModelSerializer):
    participant = serializers.SerializerMethodField(read_only=True)
    instructor = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = User
        # Django 기본 User 모델에 존재하는 필드 중 일부
        fields = (
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'last_login',  # 가장 최근 로그인 시점
            'date_joined',  # 가입 시점
            'participant',
            'instructor'
        )
        extra_kwargs = {'password': {'write_only': True}}

    def get_participant(self, user):
        if not user.participant:
            return None

        return ParticipantSerializer(user.participant, context=self.context).data

    def get_instructor(self, user):
        if not user.instructor:
            return None

        return InstructorSerializer(user.instructor, context=self.context).data

    def create(self, validated_data):
        user = super().create(validated_data)
        return user


class ParticipantSerializer(serializers.ModelSerializer):
    seminars = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = ParticipantProfile
        fields = (
            'id',
            'university',
            'accepted',
            'seminars'
        )

    def create(self, validated_data):
        university = validated_data.get('university', '')
        return ParticipantProfile(university=university)

    def update(self, instance, validated_data):
        instance.university = validated_data.get('university', '')
        return instance

    def get_seminars(self, participant):
        if not UserSeminar.objects.filter(user=User.objects.get(participant=participant), role='participant'):
            return None

        self.context['user'] = User.objects.get(participant=participant)
        queryset = map(lambda x: x.seminar, UserSeminar.objects.filter(user=User.objects.get(participant=participant), role='participant'))
        return ParticipantSeminarSerializer(queryset, many=True, context=self.context).data


class InstructorSerializer(serializers.ModelSerializer):
    charge = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = InstructorProfile
        fields = (
            'id',
            'company',
            'year',
            'charge'
        )

    def create(self, validated_data):
        company = validated_data.get('company', '')
        year = validated_data.get('year')
        return InstructorProfile(company=company, year=year)

    def update(self, instance, validated_data):
        instance.company = validated_data.get('company', '')
        instance.year = validated_data.get('year')
        instance.save()
        return instance

    def get_charge(self, instructor):
        if not UserSeminar.objects.filter(user=User.objects.get(instructor=instructor), role='instructor'):
            return None

        self.context['user'] = User.objects.get(instructor=instructor)
        query = UserSeminar.objects.get(user=User.objects.get(instructor=instructor), role='instructor').seminar
        return InstructorSeminarSerializer(query, context=self.context).data
