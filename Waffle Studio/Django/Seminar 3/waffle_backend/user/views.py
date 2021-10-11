from django.contrib.auth import authenticate, login, logout, get_user_model
from django.db import IntegrityError
from rest_framework import status, viewsets, permissions
from rest_framework.views import APIView
from rest_framework.decorators import action
from rest_framework.response import Response

from user.serializers import UserSerializer, UserLoginSerializer, UserCreateSerializer, ParticipantSerializer, InstructorSerializer, CreateParticipantProfileService

User = get_user_model()


class UserSignUpView(APIView):
    permission_classes = (permissions.AllowAny, )
    serializer_class = UserCreateSerializer

    # POST /api/v1/signup/
    # Request Body : username, password, email, role, (university, accepted), (company, year)
    def post(self, request, *args, **kwargs):
        serializer = UserCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            user, jwt_token = serializer.save()
        except IntegrityError:
            return Response(status=status.HTTP_409_CONFLICT, data='이미 존재하는 유저 이메일입니다.')

        return Response({'user': user.email, 'token': jwt_token}, status=status.HTTP_201_CREATED)


class UserLoginView(APIView):
    permission_classes = (permissions.AllowAny, )
    serializer_class = UserLoginSerializer

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

        user = request.user
        if user.is_anonymous:
            return Response(status=status.HTTP_403_FORBIDDEN, data='먼저 로그인 하세요.')

        if user.participant:
            university = request.data.get('university')
            serializer = ParticipantSerializer(user.participant, data={'university': university}, partial=True)
            serializer.is_valid(raise_exception=True)
            serializer.save()
        if user.instructor:
            company = request.data.get('company')
            year = request.data.get('year')
            serializer = InstructorSerializer(user.instructor, data={'company': company, 'year': year}, partial=True)
            serializer.is_valid(raise_exception=True)
            serializer.save()

        serializer = self.get_serializer(user, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(self.get_serializer(user).data)

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
        service = CreateParticipantProfileService(data=request.data, context={'request': request})

        status_code, data = service.execute()
        return Response(status=status_code, data=data)
