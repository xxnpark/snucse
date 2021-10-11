from rest_framework import status, viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from datetime import datetime, time

from seminar.serializers import SeminarSerializer, SimpleSeminarSerializer
from seminar.models import UserSeminar, Seminar


class SeminarViewSet(viewsets.GenericViewSet):
    permission_classes = (permissions.IsAuthenticated, )
    serializer_class = SeminarSerializer
    queryset = Seminar.objects.all()

    # POST /api/v1/seminar/ : 세미나 생성
    # Request Body : name, capacity, count, time, online
    def create(self, request):
        if request.user.is_anonymous:
            return Response(status=status.HTTP_403_FORBIDDEN, data='먼저 로그인 하세요.')

        user = request.user
        data = request.data.copy()
        if not user.instructor:
            return Response(status=status.HTTP_403_FORBIDDEN, data='세미나 진행자 자격 가진 유저만 요청할 수 있습니다.')

        name = data.get('name')
        capacity = data.get('capacity')
        count = data.get('count')
        time_ = data.get('time')
        online = data.get('online')

        if not (name and capacity and count and time):
            return Response(status=status.HTTP_400_BAD_REQUEST, data='세미나 정보가 모두 입력되지 않았습니다.')

        if not (capacity.isdigit() and int(capacity) > 0) or not (count.isdigit() and int(count) > 0):
            return Response(status=status.HTTP_400_BAD_REQUEST, data='정원 혹은 세미나 횟수가 자연수가 아닙니다.')
        data['capacity'] = int(capacity)
        data['count'] = int(count)

        try:
            valid_time = datetime.strptime(time_, '%H:%M')
            data['time'] = time(valid_time.hour, valid_time.minute)
        except ValueError:
            return Response(status=status.HTTP_400_BAD_REQUEST, data='시간 형식이 잘못되었습니다.')

        true = ['t', 'T', 'y', 'Y', 'yes', 'Yes', 'YES', 'true', 'True', 'TRUE', 'on', 'On', 'ON', '1', 1, True]
        false = ['f', 'F', 'n', 'N', 'no', 'No', 'NO', 'false', 'False', 'FALSE', 'off', 'Off', 'OFF', '0', 0, 0.0, False]
        if online in true:
            data['online'] = True
        if online in false:
            data['online'] = False

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        seminar = serializer.save()
        UserSeminar.objects.create(user=user, seminar=seminar, role='instructor')

        return Response(self.get_serializer(seminar).data, status=status.HTTP_201_CREATED)

    # PUT /api/v1/seminar/{seminar_id}/ : 세미나 수정
    # Request Body : name, capacity, count, time, online
    def update(self, request, pk):
        if request.user.is_anonymous:
            return Response(status=status.HTTP_403_FORBIDDEN, data='먼저 로그인 하세요.')

        user = request.user
        data = request.data.copy()
        if not user.instructor:
            return Response(status=status.HTTP_403_FORBIDDEN, data='세미나 진행자 자격 가진 유저만 요청할 수 있습니다.')

        try:
            seminar = Seminar.objects.get(id=pk)
        except Seminar.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND, data='seminar_id에 해당하는 세미나가 없습니다.')

        capacity = data.get('capacity')
        if capacity:
            if not (capacity.isdigit() and int(capacity) > 0):
                return Response(status=status.HTTP_400_BAD_REQUEST, data='정원이 자연수가 아닙니다.')
            data['capacity'] = int(capacity)
            if capacity < len(UserSeminar.objects.filter(seminar=seminar, role='participant', is_active=True)):
                return Response(status=status.HTTP_400_BAD_REQUEST, data='현재 세미나 참여 인원보다 정원이 적을 수 없습니다.')

        count = data.get('count')
        if count:
            if not (count.isdigit() and int(count) > 0):
                return Response(status=status.HTTP_400_BAD_REQUEST, data='횟수가 자연수가 아닙니다.')
            data['count'] = int(count)

        serializer = self.get_serializer(seminar, data=data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.update(seminar, serializer.validated_data)

        return Response(self.get_serializer(seminar).data)

    # GET /api/v1/seminar/{seminar_id}/ : 세미나 조회
    def retrieve(self, request, pk):
        try:
            seminar = Seminar.objects.get(id=pk)
            return Response(self.get_serializer(seminar).data)
        except Seminar.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND, data='존재하지 않는 세미나입니다.')

    # GET /api/v1/seminar/?name={name}&order=earliest : 여러 세미나 조회
    def list(self, request):
        param_name = request.GET.get('name')
        param_order = request.GET.get('order')
        if param_order == 'earliest':
            if param_name:
                return Response(SimpleSeminarSerializer(Seminar.objects.filter(name__contains=param_name).reverse(), many=True).data)
            else:
                return Response(SimpleSeminarSerializer(Seminar.objects.all().reverse(), many=True).data)
        else:
            if param_name:
                return Response(SimpleSeminarSerializer(Seminar.objects.filter(name__contains=param_name), many=True).data)
            else:
                return Response(SimpleSeminarSerializer(Seminar.objects.all(), many=True).data)

    # POST or DELETE
    @action(detail=True, methods=['POST', 'DELETE'])
    def user(self, request, pk):
        # POST /api/v1/seminar/{seminar_id}/user/ : 세미나 참가 (참가자 or 진행자)
        # Request Body : role
        if request.method == 'POST':
            if request.user.is_anonymous:
                return Response(status=status.HTTP_403_FORBIDDEN, data='먼저 로그인 하세요.')

            user = request.user

            try:
                seminar = Seminar.objects.get(id=pk)
            except Seminar.DoesNotExist:
                return Response(status=status.HTTP_404_NOT_FOUND, data='seminar_id에 해당하는 세미나가 없습니다.')

            role = request.data.get('role')
            if role != "participant" and role != "instructor":
                return Response(status=status.HTTP_400_BAD_REQUEST, data='올바르지 않은 역할입니다.')
            if (not user.participant and role == 'participant') or (not user.instructor and role == 'instructor'):
                return Response(status=status.HTTP_403_FORBIDDEN, data='올바르지 않은 역할입니다.')
            if user.participant and not user.participant.accepted:
                return Response(status=status.HTTP_403_FORBIDDEN, data='허용되지 않은 참가자입니다.')

            if role == 'participant':
                if len(UserSeminar.objects.filter(seminar=seminar, role=role, is_active=True)) >= seminar.capacity:
                    return Response(status=status.HTTP_400_BAD_REQUEST, data='세미나 정원이 초과되었습니다.')
                if UserSeminar.objects.filter(user=user, seminar=seminar):
                    return Response(status=status.HTTP_400_BAD_REQUEST, data='이미 해당 세미나에 참가중이거나 중도 포기하셨습니다.')

                UserSeminar.objects.create(user=user, seminar=seminar, role=role)

            if role == 'instructor':
                if UserSeminar.objects.filter(user=user, role='instructor'):
                    return Response(status=status.HTTP_400_BAD_REQUEST, data='이미 담당중인 세미나가 있습니다.')
                if UserSeminar.objects.filter(user=user, seminar=seminar):
                    return Response(status=status.HTTP_400_BAD_REQUEST, data='이미 해당 세미나에 참가중입니다.')

                UserSeminar.objects.create(user=user, seminar=seminar, role=role)

            return Response(self.get_serializer(seminar).data, status=status.HTTP_201_CREATED)

        # DELETE /api/v1/seminar/{seminar_id}/user/ : 참가자의 세미나 드랍
        if request.method == 'DELETE':
            if request.user.is_anonymous:
                return Response(status=status.HTTP_403_FORBIDDEN, data='먼저 로그인 하세요.')

            user = request.user

            try:
                seminar = Seminar.objects.get(id=pk)
            except Seminar.DoesNotExist:
                return Response(status=status.HTTP_404_NOT_FOUND, data='seminar_id에 해당하는 세미나가 없습니다.')

            if UserSeminar.objects.get(user=user, seminar=seminar).role == 'instructor':
                return Response(status=status.HTTP_403_FORBIDDEN, data='진행자는 세미나를 포기할 수 없습니다.')

            userseminar = UserSeminar.objects.get(user=user, seminar=seminar)
            userseminar.is_active = False
            userseminar.dropped_at = timezone.now()
            userseminar.save()

            return Response(self.get_serializer(seminar).data, status=status.HTTP_201_CREATED)
