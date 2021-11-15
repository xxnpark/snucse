from django.contrib.auth import get_user_model
from rest_framework import serializers, status
from django.utils import timezone
from seminar.models import Seminar, UserSeminar

User = get_user_model()

ROLE_CHOICES = (
    ('participant', 'participant'),
    ('instructor', 'instructor'),
)


class ParticipantUserProfileSerializer(serializers.ModelSerializer):
    joined_at = serializers.SerializerMethodField(read_only=True)
    is_active = serializers.SerializerMethodField(read_only=True)
    dropped_at = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = User
        fields = (
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'joined_at',
            'is_active',
            'dropped_at'
        )

    def get_joined_at(self, user):
        return user.user_seminars.get(seminar=self.context['seminar']).created_at

    def get_is_active(self, user):
        return user.user_seminars.get(seminar=self.context['seminar']).is_active

    def get_dropped_at(self, user):
        return user.user_seminars.get(seminar=self.context['seminar']).dropped_at


class InstructorUserProfileSerializer(serializers.ModelSerializer):
    joined_at = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = User
        fields = (
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'joined_at'
        )

    def get_joined_at(self, user):
        return user.user_seminars.get(seminar=self.context['seminar']).created_at


class SeminarSerializer(serializers.ModelSerializer):
    name = serializers.CharField(required=False)
    capacity = serializers.IntegerField(required=False)
    count = serializers.IntegerField(required=False)
    time = serializers.TimeField(required=False)
    online = serializers.BooleanField(required=False)
    instructors = serializers.SerializerMethodField(read_only=True)
    participants = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Seminar
        fields = (
            'id',
            'name',
            'capacity',
            'count',
            'time',
            'online',
            'instructors',
            'participants'
        )

    def validate_capacity(self, value):
        if value <= 0:
            raise serializers.ValidationError('정원이 자연수가 아닙니다.')
        return value

    def validate_count(self, value):
        if value <= 0:
            raise serializers.ValidationError('세미나 횟수가 자연수가 아닙니다.')
        return value

    def get_instructors(self, seminar):
        queryset = map(lambda us: us.user, seminar.user_seminars.filter(role='instructor'))
        return InstructorUserProfileSerializer(queryset, many=True, context={'seminar': seminar}).data

    def get_participants(self, seminar):
        queryset = map(lambda us: us.user, seminar.user_seminars.filter(role='participant'))
        return ParticipantUserProfileSerializer(queryset, many=True, context={'seminar': seminar}).data

    def create(self, validated_data):
        user = self.context['request'].user
        if user.is_anonymous:
            return status.HTTP_403_FORBIDDEN, '먼저 로그인 하세요.'
        if not user.instructor:
            return status.HTTP_403_FORBIDDEN, '세미나 진행자 자격을 가진 유저만 요청할 수 있습니다.'

        name = self.validated_data.get('name')
        capacity = self.validated_data.get('capacity')
        count = self.validated_data.get('count')
        time = self.validated_data.get('time')

        if not (name and capacity and count and time):
            return status.HTTP_400_BAD_REQUEST, '세미나 정보가 모두 입력되지 않았습니다.'

        seminar = super().create(validated_data)
        UserSeminar.objects.create(user=user, seminar=seminar, role='instructor')

        return status.HTTP_201_CREATED, SeminarSerializer(seminar).data

    def update(self, instance, validated_data):
        user = self.context['request'].user
        if user.is_anonymous:
            return status.HTTP_403_FORBIDDEN, '먼저 로그인 하세요.'
        if not user.user_seminars.filter(seminar=instance, role='instructor').exists():
            return status.HTTP_403_FORBIDDEN, '세미나 진행자 자격을 가진 유저만 요청할 수 있습니다.'

        capacity = validated_data.get('capacity')
        if capacity and capacity < instance.user_seminars.filter(role='participant', is_active=True).count():
            return status.HTTP_400_BAD_REQUEST, '현재 세미나 참여 인원보다 정원이 적을 수 없습니다.'

        super().update(instance, validated_data)

        return status.HTTP_200_OK, SeminarSerializer(instance).data


class SimpleSeminarSerializer(SeminarSerializer):
    participant_count = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Seminar
        fields = (
            'id',
            'name',
            'instructors',
            'participant_count'
        )

    def get_participant_count(self, seminar):
        return seminar.user_seminars.filter(role='participant', is_active=True).count()
        # return len(UserSeminar.objects.filter(seminar=seminar, role='participant', is_active=True)) 차이점 아직 모르겠음


class RegisterSeminarService(serializers.Serializer):
    role = serializers.ChoiceField(choices=ROLE_CHOICES,)

    def execute(self):
        self.is_valid(raise_exception=True)

        user = self.context['request'].user

        if user.is_anonymous:
            return status.HTTP_403_FORBIDDEN, '먼저 로그인 하세요.'
        if user.participant and not user.participant.accepted:
            return status.HTTP_403_FORBIDDEN, '허용되지 않은 참가자입니다.'

        seminar_id = self.context['pk']

        try:
            seminar = Seminar.objects.get(id=seminar_id)
        except Seminar.DoesNotExist:
            return status.HTTP_404_NOT_FOUND, 'seminar_id에 해당하는 세미나가 없습니다.'

        if seminar.user_seminars.filter(user=user):
            return status.HTTP_400_BAD_REQUEST, '이미 해당 세미나에 참가중이거나 중도 포기하셨습니다.'

        role = self.validated_data['role']

        if role == 'participant':
            if not user.participant:
                return status.HTTP_403_FORBIDDEN, '잘못된 role입니다.'
            if seminar.user_seminars.filter(role=role, is_active=True).count() >= seminar.capacity:
                return status.HTTP_400_BAD_REQUEST, '세미나 정원이 초과되었습니다.'

        if role == 'instructor':
            if not user.instructor:
                return status.HTTP_403_FORBIDDEN, '잘못된 role입니다.'
            if user.user_seminars.filter(role='instructor'):
                return status.HTTP_400_BAD_REQUEST, '이미 담당중인 세미나가 있습니다.'

        UserSeminar.objects.create(user=user, seminar=seminar, role=role)

        return status.HTTP_201_CREATED, SeminarSerializer(seminar).data


class DropSeminarService(serializers.Serializer):
    def execute(self):
        user = self.context['request'].user
        if user.is_anonymous:
            return status.HTTP_403_FORBIDDEN, '먼저 로그인 하세요.'

        seminar_id = self.context['pk']

        try:
            seminar = Seminar.objects.get(id=seminar_id)
        except Seminar.DoesNotExist:
            return status.HTTP_404_NOT_FOUND, 'seminar_id에 해당하는 세미나가 없습니다.'

        try:
            userseminar = seminar.user_seminars.get(user=user)
        except UserSeminar.DoesNotExist:
            return status.HTTP_200_OK, SeminarSerializer(seminar).data

        if userseminar.role == 'instructor':
            return status.HTTP_403_FORBIDDEN, '진행자는 세미나를 포기할 수 없습니다.'

        userseminar.is_active = False
        userseminar.dropped_at = timezone.now()
        userseminar.save()

        return status.HTTP_200_OK, SeminarSerializer(seminar).data
