from django.contrib.auth import get_user_model
from rest_framework import serializers
from seminar.models import Seminar, UserSeminar

User = get_user_model()

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
        return UserSeminar.objects.get(user=self.context['user'], seminar=seminar).created_at
    def get_is_active(self, seminar):
        return UserSeminar.objects.get(user=self.context['user'], seminar=seminar).is_active
    def get_dropped_at(self, seminar):
        return UserSeminar.objects.get(user=self.context['user'], seminar=seminar).dropped_at


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
        return UserSeminar.objects.get(user=self.context['user'], seminar=seminar).created_at


class ParticipantUserSerializer(serializers.ModelSerializer):
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
        return UserSeminar.objects.get(user=user, seminar=self.context['seminar']).created_at

    def get_is_active(self, user):
        return UserSeminar.objects.get(user=user, seminar=self.context['seminar']).is_active

    def get_dropped_at(self, user):
        return UserSeminar.objects.get(user=user, seminar=self.context['seminar']).dropped_at


class InstructorUserSerializer(serializers.ModelSerializer):
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
        return UserSeminar.objects.get(user=user, seminar=self.context['seminar']).created_at


class SeminarSerializer(serializers.ModelSerializer):
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

    def get_instructors(self, seminar):
        queryset = map(lambda x: x.user, UserSeminar.objects.filter(seminar=seminar, role='instructor'))
        self.context['seminar'] = seminar
        return InstructorUserSerializer(queryset, many=True, context=self.context).data

    def get_participants(self, seminar):
        queryset = map(lambda x: x.user, UserSeminar.objects.filter(seminar=seminar, role='participant'))
        self.context['seminar'] = seminar
        return ParticipantUserSerializer(queryset, many=True, context=self.context).data

    def create(self, validated_data):
        name = validated_data.get('name')
        capacity = validated_data.get('capacity')
        count = validated_data.get('count')
        time = validated_data.get('time')
        online = validated_data.get('online', True)
        user = Seminar.objects.create(name=name, capacity=capacity, count=count, time=time, online=online)
        return user


class SimpleSeminarSerializer(serializers.ModelSerializer):
    instructors = serializers.SerializerMethodField(read_only=True)
    participant_count = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Seminar
        fields = (
            'id',
            'name',
            'instructors',
            'participant_count'
        )

    def get_instructors(self, seminar):
        queryset = map(lambda x: x.user, UserSeminar.objects.filter(seminar=seminar, role='instructor'))
        self.context['seminar'] = seminar
        return InstructorUserSerializer(queryset, many=True, context=self.context).data

    def get_participant_count(self, seminar):
        return len(UserSeminar.objects.filter(seminar=seminar, role='participant', is_active=True))
        # return len(seminar.seminar.filter(role='participant', is_active=True)) 차이점 아직 모르겠음
