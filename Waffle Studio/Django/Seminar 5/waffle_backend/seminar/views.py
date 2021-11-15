from rest_framework import status, viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response

from seminar.serializers import SeminarSerializer, SimpleSeminarSerializer, RegisterSeminarService, DropSeminarService
from seminar.models import UserSeminar, Seminar


class SeminarViewSet(viewsets.GenericViewSet):
    permission_classes = (permissions.IsAuthenticated, )
    serializer_class = SeminarSerializer
    queryset = Seminar.objects.all()

    # POST /api/v1/seminar/ : 세미나 생성
    # Request Body : name, capacity, count, time, online
    def create(self, request):
        serializer = self.get_serializer(data=request.data, context={'request': request}, partial=True)
        serializer.is_valid(raise_exception=True)
        status_code, data = serializer.save()
        return Response(status=status_code, data=data)

    # PUT /api/v1/seminar/{seminar_id}/ : 세미나 수정
    # Request Body : name, capacity, count, time, online
    def update(self, request, pk=None):
        try:
            seminar = Seminar.objects.get(id=pk)
        except Seminar.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND, data='존재하지 않는 세미나입니다.')

        serializer = self.get_serializer(seminar, data=request.data, context={'request': request}, partial=True)
        serializer.is_valid(raise_exception=True)
        status_code, data = serializer.save()
        return Response(status=status_code, data=data)

    # GET /api/v1/seminar/{seminar_id}/ : 세미나 조회
    def retrieve(self, request, pk=None):
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
                return Response(SimpleSeminarSerializer(self.queryset.filter(name__contains=param_name).order_by('-id'), many=True).data)
            else:
                return Response(SimpleSeminarSerializer(self.queryset.order_by('-id'), many=True).data)
        else:
            if param_name:
                return Response(SimpleSeminarSerializer(Seminar.objects.filter(name__contains=param_name), many=True).data)
            else:
                return Response(SimpleSeminarSerializer(self.queryset, many=True).data)

    # POST or DELETE
    @action(detail=True, methods=['POST', 'DELETE'])
    def user(self, request, pk=None):
        # POST /api/v1/seminar/{seminar_id}/user/ : 세미나 참가 (참가자 or 진행자)
        # Request Body : role
        if request.method == 'POST':
            service = RegisterSeminarService(data=request.data, context={'request': request, 'pk': pk})
            status_code, data = service.execute()
            return Response(status=status_code, data=data)

        # DELETE /api/v1/seminar/{seminar_id}/user/ : 참가자의 세미나 드랍
        if request.method == 'DELETE':
            service = DropSeminarService(context={'request': request, 'pk': pk})
            status_code, data = service.execute()
            return Response(status=status_code, data=data)
