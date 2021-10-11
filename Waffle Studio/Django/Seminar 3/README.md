# waffle-rookies-19.5-backend-2

와플스튜디오 19.5기 루키 Django 세미나 3 [과제](https://github.com/wafflestudio/19.5-rookies/tree/master/django/seminar3/assignment3)

> 학교 과제와 백신 2차 접종 등으로 제시간에 과제를 완료하지 못해 grace day 1일 사용합니다.
> 
> 자유과제는 아직 하지 못했습니다.

---
### Test를 진행하면서 과제 2에서 수정한 사항

* 전반적으로 `views.py`를 간소화하고 대부분의 API 구현 내용과 validation은 `serializers.py`로 옮겼다. 

#### POST /api/v1/signup/
* `ParticipantProfile`, `InstructorProfile` model을 만드는 `ParticipantSerializer`, `InstructorSerializer`에 university, company, year 입력의 유효성 확인을 `validate_field` method로 관리하였다.
    ```python
    class ParticipantSerializer(serializers.ModelSerializer):
        university = serializers.CharField(required=False, allow_null=True)
    
        def validate_university(self, value):
            return value or ''
    
    class InstructorSerializer(serializers.ModelSerializer):
        company = serializers.CharField(required=False, allow_null=True)
        year = serializers.IntegerField(required=False, allow_null=True)
    
        def validate_company(self, value):
            return value or ''
    
        def validate_year(self, value):
            if value and value <= 0:
                raise serializers.ValidationError('올바르지 않은 연도 형식입니다.')
            return value
    ```

#### GET /api/v1/user/{user_id}/
* `ParticipantSeminarSerializer`, `InstructorSeminarSerializer`가 `seminar/serializers.py`에 정의되어있었는데, `user/serializers.py`로 옮겼다. `ParticipantSerializer`, `InstuctorSerializer`에서 사용자의 참여 세미나 정보를 가져오는 데에만 이용되기 때문에 `seminar/` 내에 이용될 일이 없었다.

#### POST /api/v1/seminar/
* capacity, count, time, online 필드의 입력이 유효한 정수, 시각, 참/거짓 값인지 직접 확인하지 않고, `serializers.IntegerField`, `serializers.TimeField`, `serializers.BooleanField`를 이용하여 자동으로 검사하도록 하였다. 이 경우 잘못된 입력인 경우 `400 Bad Request` 에러를 반환하였다.
    ```python
    class SeminarSerializer(serializers.ModelSerializer):
        name = serializers.CharField(required=False)
        capacity = serializers.IntegerField(required=False)
        count = serializers.IntegerField(required=False)
        time = serializers.TimeField(required=False)
        online = serializers.BooleanField(required=False)
    
        def validate_capacity(self, value):
            if value <= 0:
                raise serializers.ValidationError('정원이 자연수가 아닙니다.')
            return value
    
        def validate_count(self, value):
            if value <= 0:
                raise serializers.ValidationError('세미나 횟수가 자연수가 아닙니다.')
            return value
    ```
* name, capacity, count, time 입력은 필수로 입력되어야했기 때문에 `required=True`와 같이 처리해주려고 했지만, `SeminarSerializer`는 세미나 수정 시에도 이용했기에 그렇게 할 수 없었다. `.create()` method에서 직접 처리해주었다.

#### GET /api/v1/seminar/
* 원래는 `&order=earliest` param을 넘기는 경우 `queryset.reverse()`를 이용하여 순서를 뒤집어주었지만, test를 통과하지 못했다. [문서](https://docs.djangoproject.com/en/3.2/ref/models/querysets/#reverse)를 찾아본 결과 default ordering이 있는 모델을 query할 때만 유효하게 적용된다 하여 `queryset.order_by('-id')`을 이용하였다.

---
### Coverage 내역
```zsh
(venv) $ coverage run manage.py test
(venv) $ coverage report            
Name                                                         Stmts   Miss  Cover
--------------------------------------------------------------------------------
manage.py                                                       12      2    83%
seminar/__init__.py                                              0      0   100%
seminar/admin.py                                                 1      0   100%
seminar/apps.py                                                  4      0   100%
seminar/migrations/0001_initial.py                               9      0   100%
seminar/migrations/0002_auto_20210926_0418.py                    4      0   100%
seminar/migrations/0003_auto_20211006_1829.py                    6      0   100%
seminar/migrations/__init__.py                                   0      0   100%
seminar/models.py                                               22      0   100%
seminar/serializers.py                                         133      4    97%
seminar/tests.py                                               394      0   100%
seminar/urls.py                                                  6      0   100%
seminar/views.py                                                49      1    98%
survey/__init__.py                                               0      0   100%
survey/admin.py                                                  4      0   100%
survey/apps.py                                                   3      0   100%
survey/management/__init__.py                                    0      0   100%
survey/management/commands/__init__.py                           0      0   100%
survey/migrations/0001_initial.py                                6      0   100%
survey/migrations/0002_auto_20210910_1509.py                     6      0   100%
survey/migrations/__init__.py                                    0      0   100%
survey/models.py                                                19      0   100%
survey/serializers.py                                           26      8    69%
survey/tests.py                                                  7      0   100%
survey/urls.py                                                   8      0   100%
survey/views.py                                                 42     21    50%
user/__init__.py                                                 0      0   100%
user/admin.py                                                    1      0   100%
user/apps.py                                                     3      0   100%
user/migrations/0001_initial.py                                  7      0   100%
user/migrations/0002_auto_20210925_0916.py                       6      0   100%
user/migrations/0003_remove_participantprofile_seminars.py       4      0   100%
user/migrations/0004_remove_instructorprofile_charge.py          4      0   100%
user/migrations/0005_auto_20211006_1830.py                       4      0   100%
user/migrations/__init__.py                                      0      0   100%
user/models.py                                                  51      8    84%
user/serializers.py                                            177     23    87%
user/tests.py                                                  214      0   100%
user/urls.py                                                     7      0   100%
user/views.py                                                   62      6    90%
waffle_backend/__init__.py                                       0      0   100%
waffle_backend/settings.py                                      30      3    90%
waffle_backend/urls.py                                           9      2    78%
--------------------------------------------------------------------------------
TOTAL                                                         1340     78    94%

```