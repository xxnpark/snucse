from abc import ABC
from django.contrib.auth import get_user_model, authenticate
from django.contrib.auth.models import update_last_login
from django.db import transaction
from rest_framework import serializers, status
from rest_framework_jwt.settings import api_settings
from user.models import ParticipantProfile, InstructorProfile
from seminar.models import Seminar, UserSeminar

ROLE_CHOICES = (
    ('participant', 'participant'),
    ('instructor', 'instructor'),
)

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
    first_name = serializers.CharField(required=False)
    last_name = serializers.CharField(required=False)
    role = serializers.ChoiceField(choices=ROLE_CHOICES, required=True)
    university = serializers.CharField(required=False)
    accepted = serializers.BooleanField(required=False)
    company = serializers.CharField(required=False)
    year = serializers.IntegerField(required=False)

    def validate(self, data):
        first_name = data.get('first_name')
        last_name = data.get('last_name')

        if bool(first_name) ^ bool(last_name):
            raise serializers.ValidationError("성과 이름 중에 하나만 입력할 수 없습니다.")
        if first_name and last_name and not (first_name.isalpha() and last_name.isalpha()):
            raise serializers.ValidationError("이름에 숫자가 들어갈 수 없습니다.")

        return data

    @transaction.atomic
    def create(self, validated_data):
        email = validated_data.get('email')
        username = validated_data.get('username')
        password = validated_data.get('password')
        first_name = validated_data.get('first_name')
        last_name = validated_data.get('last_name')

        user = User.objects.create_user(email=email, username=username, password=password, first_name=first_name, last_name=last_name)

        role = validated_data.get('role')

        if role == "participant":
            university = validated_data.get('university')
            accepted = validated_data.get('accepted', True)
            serializer = ParticipantSerializer(data={'university': university, 'accepted': accepted})
            serializer.is_valid(raise_exception=True)
            user.participant = serializer.save()
            user.save()

        if role == "instructor":
            company = validated_data.get('company')
            year = validated_data.get('year')
            serializer = InstructorSerializer(data={'company': company, 'year': year})
            serializer.is_valid(raise_exception=True)
            user.instructor = serializer.save()
            user.save()

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
    first_name = serializers.CharField(required=False)
    last_name = serializers.CharField(required=False)
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
            'instructor',
        )
        extra_kwargs = {'password': {'write_only': True}}

    def validate(self, data):
        first_name = data.get('first_name')
        last_name = data.get('last_name')

        if bool(first_name) ^ bool(last_name):
            raise serializers.ValidationError("성과 이름 중에 하나만 입력할 수 없습니다.")
        if first_name and last_name and not (first_name.isalpha() and last_name.isalpha()):
            raise serializers.ValidationError("이름에 숫자가 들어갈 수 없습니다.")

        return data

    def get_participant(self, user):
        if not user.participant:
            return None

        return ParticipantSerializer(user.participant, context=self.context).data

    def get_instructor(self, user):
        if not user.instructor:
            return None

        return InstructorSerializer(user.instructor, context=self.context).data

    def create(self, validated_data):
        return super().create(validated_data)

    def update(self, instance, validated_data):
        return super().update(instance, validated_data)


class ParticipantSerializer(serializers.ModelSerializer):
    university = serializers.CharField(required=False, allow_null=True)
    accepted = serializers.BooleanField(required=False, allow_null=True)
    seminars = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = ParticipantProfile
        fields = (
            'id',
            'university',
            'accepted',
            'seminars'
        )

    def validate_university(self, value):
        return value or ''

    def get_seminars(self, participant):
        self.context['user'] = User.objects.get(participant=participant)
        queryset = map(lambda x: x.seminar, UserSeminar.objects.filter(user=User.objects.get(participant=participant), role='participant'))
        return ParticipantSeminarSerializer(queryset, many=True, context=self.context).data

    def create(self, validated_data):
        return super().create(validated_data)

    def update(self, instance, validated_data):
        return super().update(instance, validated_data)


class ParticipantSeminarSerializer(serializers.ModelSerializer):
    joined_at = serializers.SerializerMethodField(read_only=True)
    is_active = serializers.SerializerMethodField(read_only=True)
    dropped_at = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Seminar
        fields = (
            'id',
            'name',
            'joined_at',
            'is_active',
            'dropped_at'
        )

    def get_joined_at(self, seminar):
        return seminar.user_seminars.get(user=self.context['user']).created_at

    def get_is_active(self, seminar):
        return seminar.user_seminars.get(user=self.context['user']).is_active

    def get_dropped_at(self, seminar):
        return seminar.user_seminars.get(user=self.context['user']).dropped_at


class InstructorSerializer(serializers.ModelSerializer):
    company = serializers.CharField(required=False, allow_null=True)
    year = serializers.IntegerField(required=False, allow_null=True)
    charge = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = InstructorProfile
        fields = (
            'id',
            'company',
            'year',
            'charge'
        )

    def validate_company(self, value):
        return value or ''

    def validate_year(self, value):
        if value and value <= 0:
            raise serializers.ValidationError('올바르지 않은 연도 형식입니다.')
        return value

    def get_charge(self, instructor):
        if not UserSeminar.objects.filter(user=User.objects.get(instructor=instructor), role='instructor'):
            return None

        self.context['user'] = User.objects.get(instructor=instructor)
        queryset = map(lambda x: x.seminar, UserSeminar.objects.filter(user=User.objects.get(instructor=instructor), role='instructor'))
        # query = UserSeminar.objects.get(user=User.objects.get(instructor=instructor), role='instructor').seminar
        return InstructorSeminarSerializer(queryset, many=True, context=self.context).data or []

    def create(self, validated_data):
        return super().create(validated_data)

    def update(self, instance, validated_data):
        return super().update(instance, validated_data)


class InstructorSeminarSerializer(serializers.ModelSerializer):
    joined_at = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Seminar
        fields = (
            'id',
            'name',
            'joined_at',
        )

    def get_joined_at(self, seminar):
        return seminar.user_seminars.get(user=self.context['user']).created_at


class CreateParticipantProfileService(serializers.Serializer):
    university = serializers.CharField(required=False, allow_null=True)
    accepted = serializers.BooleanField(required=False, allow_null=True)

    def execute(self):
        self.is_valid(raise_exception=True)

        user = self.context['request'].user

        if user.is_anonymous:
            return status.HTTP_403_FORBIDDEN, '먼저 로그인 하세요.'
        if user.participant:
            return status.HTTP_400_BAD_REQUEST, '이미 참여자로 지정되었습니다.'

        university = self.validated_data.get('university')
        accepted = self.validated_data.get('accepted', True)

        serializer = ParticipantSerializer(data={'university': university, 'accepted': accepted})
        serializer.is_valid(raise_exception=True)
        user.participant = serializer.save()
        user.save()
        return status.HTTP_201_CREATED, UserSerializer(user).data
