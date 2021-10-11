import json
from django.http import JsonResponse, HttpResponseNotAllowed, Http404, HttpResponseBadRequest
from django.shortcuts import get_object_or_404

from survey.models import SurveyResult, OperatingSystem
from survey.serializers import serialize_survey_result, serialize_os


def get_survey_results(request):
    if request.method == 'GET':
        params = request.GET.get('os')
        if params:
            try:
                survey_results = list(map(lambda result: serialize_survey_result(result), OperatingSystem.objects.get(name=params).surveys.all()))
            except OperatingSystem.DoesNotExist:
                return HttpResponseBadRequest("OS does not exist.")
        else:
            survey_results = list(map(lambda result: serialize_survey_result(result), SurveyResult.objects.all()))
        return JsonResponse({"surveys": survey_results}, status=200)
    else:
        return HttpResponseNotAllowed(['GET', ])


def get_survey(request, survey_id):
    if request.method == 'GET':
        survey = get_object_or_404(SurveyResult, id=survey_id)
        return JsonResponse(serialize_survey_result(survey))
    else:
        return HttpResponseNotAllowed(['GET', ])


def get_os_all(request):
    if request.method == 'GET':
        os_all = list(map(lambda os: serialize_os(os), OperatingSystem.objects.all()))
        return JsonResponse({"os": os_all}, status=200)
    else:
        return HttpResponseNotAllowed(['GET', ])


def get_os(request, os_id):
    if request.method == 'GET':
        try:
            survey = OperatingSystem.objects.get(id=os_id)
            return JsonResponse(serialize_os(survey), status=200)
        except OperatingSystem.DoesNotExist:
            raise Http404("OS ID does not exist.")
    else:
        return HttpResponseNotAllowed(['GET', ])