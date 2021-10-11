from django.contrib.auth import authenticate, login, logout, get_user_model
from django.db import IntegrityError
from rest_framework import status, viewsets, permissions
from rest_framework.views import APIView
from rest_framework.decorators import action
from rest_framework.response import Response

from user.serializers import UserSerializer, UserLoginSerializer, UserCreateSerializer, ParticipantSerializer, InstructorSerializer
from user.models import ParticipantProfile, InstructorProfile

User = get_user_model()


class UserSignUpView(APIView):
    permission_classes = (permissions.AllowAny, )

    # POST /api/v1/signup/
    # Request Body : username, password, email, role, (university, accepted), (company, year)
    def post(self, request, *args, **kwargs):
        role = request.data.get('role')
        if role != "participant" and role != "instructor":
            return Response(status=status.HTTP_400_BAD_REQUEST, data='올바르지 않은 역할입니다.')

        serializer = UserCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            user, jwt_token = serializer.save()
        except IntegrityError:
            return Response(status=status.HTTP_409_CONFLICT, data='이미 존재하는 유저 이메일입니다.')

        if role == "participant":
            true = ['t', 'T', 'y', 'Y', 'yes', 'Yes', 'YES', 'true', 'True', 'TRUE', 'on', 'On', 'ON', '1', 1, True]
            false = ['f', 'F', 'n', 'N', 'no', 'No', 'NO', 'false', 'False', 'FALSE', 'off', 'Off', 'OFF', '0', 0, 0.0, False]
            university = request.data.get('university', '')
            accepted = request.data.get('accepted', True)
            if accepted in true:
                accepted = True
            if accepted in false:
                accepted = False
            user.participant = ParticipantProfile.objects.create(university=university, accepted=accepted)
            user.save()

        if role == "instructor":
            company = request.data.get('company', '')
            year = request.data.get('year')
            if year and not (year.isdigit() and int(year) > 0):
                return Response(status=status.HTTP_400_BAD_REQUEST, data='올바르지 않은 연도 형식입니다.')
            user.instructor = InstructorProfile.objects.create(company=company, year=int(year))
            user.save()

        return Response({'user': user.email, 'token': jwt_token}, status=status.HTTP_201_CREATED)


class UserLoginView(APIView):
    permission_classes = (permissions.AllowAny, )

    # POST /api/v1/login/
    def post(self, request):

        serializer = UserLoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        token = serializer.validated_data['token']

        return Response({'success': True, 'token': token}, status=status.HTTP_200_OK)


class UserViewSet(viewsets.GenericViewSet):
    permission_classes = (permissions.IsAuthenticated, )
    serializer_class = UserSerializer
    queryset = User.objects.all()

    # PUT /api/v1/user/me/ : 수정
    # Request Body : (university), (company, year)
    def update(self, request, pk=None):
        if pk != 'me':
            return Response(status=status.HTTP_403_FORBIDDEN, data='다른 유저 정보를 수정할 수 없습니다.')

        if request.user.is_anonymous:
            return Response(status=status.HTTP_403_FORBIDDEN, data='먼저 로그인 하세요.')

        user = request.user
        data = request.data.copy()
        data.pop('accepted', None)

        serializer = self.get_serializer(user, data=data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.update(user, serializer.validated_data)

        if user.participant:
            serializer = ParticipantSerializer(user.participant, data=data)
            serializer.is_valid(raise_exception=True)
            serializer.save()
        if user.instructor:
            year = data.get('year')
            if year and not (year.isdigit() and int(year) > 0):
                return Response(status=status.HTTP_400_BAD_REQUEST, data='올바르지 않은 연도 형식입니다.')
            data['year'] = int(year)
            serializer = InstructorSerializer(user.instructor, data=data)
            serializer.is_valid(raise_exception=True)
            serializer.save()

        return Response(status=status.HTTP_200_OK)

    # GET /api/v1/user/{user_id}/ : 조회
    def retrieve(self, request, pk=None):
        if request.user.is_anonymous:
            return Response(status=status.HTTP_403_FORBIDDEN, data='먼저 로그인 하세요.')

        user = request.user if pk == 'me' else self.get_object()
        return Response(self.get_serializer(user).data)

    # POST /api/v1/user/participant/ : 참여자 등록
    # Request Body : university, accepted
    @action(detail=False, methods=['POST'])
    def participant(self, request):
        if request.user.is_anonymous:
            return Response(status=status.HTTP_403_FORBIDDEN, data='먼저 로그인 하세요.')

        user = request.user

        if user.participant:
            return Response(status=status.HTTP_400_BAD_REQUEST, data='이미 참여자로 지정되었습니다.')

        true = ['t', 'T', 'y', 'Y', 'yes', 'Yes', 'YES', 'true', 'True', 'TRUE', 'on', 'On', 'ON', '1', 1, True]
        false = ['f', 'F', 'n', 'N', 'no', 'No', 'NO', 'false', 'False', 'FALSE', 'off', 'Off', 'OFF', '0', 0, 0.0, False]
        university = request.data.get('university', '')
        accepted = request.data.get('accepted', True)
        if accepted in true:
            accepted = True
        if accepted in false:
            accepted = False
        user.participant = ParticipantProfile.objects.create(university=university, accepted=accepted)
        user.save()

        return Response(self.get_serializer(user).data, status=status.HTTP_201_CREATED)
