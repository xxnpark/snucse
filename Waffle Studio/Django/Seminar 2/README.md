# waffle-rookies-19.5-backend-2

와플스튜디오 19.5기 루키 Django 세미나 2 [과제](https://github.com/wafflestudio/19.5-rookies/tree/master/django/seminar2/assignment2)

> 원래 여러번에 나누어 commit을 해두었는데, `main` branch를 skeleton code와 동일하게 안했어서 거기에 또 commit하면 `workspace`에서 merge할 때 꼬이고 이런저런 문제 생길까봐 새로 directory를 만들었습니다.

[`user/`](./waffle_backend/user/), [`seminar/`](./waffle_backend/seminar/) 폴더의 `serializers.py`, `models.py`, `views.py`, `urls.py` 4가지 파일을 수정하여 과제에서 요구하는 API를 개발하였다.

`Django Debug Toolbar` 이용 시에는 [#463](https://github.com/wafflestudio/19.5-rookies/issues/463)에 따라 `settings.py`를 바꾸고, `settings.py`와 `urls.py`에 skeleton code에서는 `if DEBUG_TOOLBAR:`로 되어있던 부분을 `if DEBUG:`로 바꿔주었으며, 각 `views.py`의 ViewSet에서 `permission_classes = (permissions.IsAuthenticated, )` 부분을 주석처리해주었다.

---
### 부족한 점, 아직 궁금한 점 🥲
* `model.objects.create(**data)`, `serializer(data=data)`, `self.get_serialzier(data=data)`를 다르게 사용하는 이유가 무엇이고 각각 언제 사용하는지가 이해가 잘 안되었다. 이번 과제에서는 무작위로 혼용해서 쓴 것 같다. 또 `get_serializer()`는 하나의 ViewSet에서 하나의 serializer만 지칭할 수 있는 것 같은데, 다양하게 쓸 수는 없는건지 모르겠다.
* 비슷한 필드를 가진 serializer를 전부 따로 정의해서 사용해야되나? 이번 과제의 `SeminarSerializer`와 `SimpleSeminarSerializer`와 같은 경우가 그렇다. 큰 쪽에서 작은 serializer를 상속받는 방법은 가능할지 궁금하다.
* ~~HTTP response body에 `serializer.data`를 넘길 때, 반환에만 이용되는 field를 serializer가 사용하는 model의 field에도 꼭 추가해야 하나? 이번 과제의 `User`, `Seminar` model이 너무 불필요하게 늘어진 것 같다.~~ **_해결완료_** _`serializers.SerializerMethodField()`를 쓸때는 굳이 model에 추가를 안해도 되는거였다._
* serializer에서 model의 field를 가져올 때, 모든 field를 가져오지 않을 것인데도 DB에서는 모든 column의 정보를 가져온다. 
  * 이로 인해 `Django Debug Toolbar`을 이용하여 `GET /api/v1/seminar/` API의 query를 살펴보았을 때, 실제 response body로 반환할 값보다 DB에서 SELECt하는 값이 불필요하게 많아 데이터베이스를 비효율적으로 사용하는 것 같다.
  ![body](./results/body.png)
  ![query](./results/query.png)
* `models.ForeignKey()`로 연결된 model 관계에서 역참조를 할 때, `related_name`이 익숙하지 않아서 한번도 이용하지 않았다. 전부 `model.objects.get()`, `models.objects.filter()`로만 참조했다. 근데 다음과 같은 model 관계에서 `seminar.name`이 `'Django'`인 `UserSeminar` model을 모두 가져오는 아래 두 방법이 무엇이 다른지 잘 모르겠다.
  ```python
  class UserSeminar(models.Model):
      seminar = models.ForeignKey(Seminar, related_name='seminar', on_delete=models.CASCADE)
  
  class Seminar(models.Model):
      name = models.CharField(max_length=20)
  ```
  ```python
  django = Seminar.objects.get(name='Django')
  objects = UserSeminar.objects.filter(seminar=django) # related_name 사용안함
  objects = django.seminar.all() # related_name 사용함
  ```
  * 코드에 적용해서 `Django Debug Toolbar`로 봐도 `seminar` 조건으로 먼저 필터링하는지 나중에 필터링하는지의 차이일 뿐, 큰 차이는 잘 모르겠다.
  ![filter](./results/querywithFilter.png)
  ![related_name](./results/querywithRelatedName.png)
* serializer 내에서 `create()`, `update()` method를 정의하면 `serializer.save()`를 이용할 수 있다는데, 두 메소드의 정확한 사용 방법, 정의를 하고 써야하는건지 아님 부모 클래스에서 미리 정의되어있는건지 아직 모르겠다. 그냥 정의를 안해도 `serializer.create()`, `serializer.update()`처럼 쓰기도 하는거같은데 `serializer.save()`는 에러가 났던 것으로 기억한다.

시간이 없어서 [공식](https://docs.djangoproject.com/en/3.2/)[문서](https://www.django-rest-framework.org/)를 대충 읽고 과제 해결만 급하게 한 탓에 까먹기 전에 더 공부해야할 것들을 기록해두었다. 조금만 더 읽으면 다 알 수 있는 내용들인 것 같으니 열심히 읽어야겠다...

뒤늦게 알게되었는데 작년 세미나의 [#176](https://github.com/wafflestudio/18.5-rookies/issues/176)에 위 궁금증에 대한 답변이 많이 있다. 이것도 읽어봐야겠다...

---
### 실행 모습
* `POST /api/v1/signup/`, `POST /api/v1/login/`을 이용한 진행자 가입, 로그인

![01](./results/01.instructorsignup.png)
![02](./results/02.instructorlogin.png)
* 로그인 결과 데이터베이스에 InstructorProfile 모델이 등록됨

![03](./results/03.instructorsprofile.png)
* `PUT /api/v1/user/me/`을 이용한 진행자 정보 변경

![04](./results/04.changeinstructorinfo.png)
* 데이터베이스에도 적용됨

![05](./results/05.instructorinfochanged.png)
* `POST /api/v1/seminar`을 이용하여 세미나 생성

![06](./results/06.makeseminar.png)
* 데이터베이스에도 적용됨

![07](./results/07.seminarmade.png)
![08](./results/08.userseminarmade.png)
* `PUT /api/v1/seminar/{seminar_id}/`을 이용하여 세미나 정보 수정

![09](./results/09.changeseminarinfo.png)
* `POST /api/v1/signup/`, `POST /api/v1/login/`을 이용한 참여자 가입, 로그인

![10](./results/10.participantsignup.png)
![11](./results/11.participantlogin.png)
* `POST /api/v1/seminar/{seminar_id}/user/`을 이용하여 참여자 세미나 참여

![12](./results/12.participantjoinseminar.png)
* `DELETE /api/v1/seminar/{seminar_id}/user/`을 이용하여 참여자 세미나 중도 포기

![13](./results/13.participantdropseminar.png)
* `POST /api/v1/user/participant/`을 이용하여 진행자의 참여자 등록

![14](./results/14.instructoraddparticipant.png)
* 데이터베이스에도 적용됨

![15](./results/15.participantadded.png)
* `POST /api/v1/seminar/{seminar_id}/user/`을 이용하여 기존 진행자였던 참여자의 세미나 참여

![16](./results/16.instructorparticipantjoinseminar.png)
* 데이터베이스에도 적용됨

![17](./results/17.userseminarmade.png)
* `POST /api/v1/seminar`을 이용하여 세미나 추가 생성

![18](./results/18.makeseminar.png)
* `GET /api/v1/seminar/`을 이용하여 세미나 목록 반환

![19](./results/19.printseminars.png)
* `GET /api/v1/user/{user_id}/`을 이용하여 특정 유저 정보 반환

![20](./results/20.printuserinfo.png)