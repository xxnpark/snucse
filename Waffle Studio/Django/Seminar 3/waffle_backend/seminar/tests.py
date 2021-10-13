from factory.django import DjangoModelFactory

from django.contrib.auth import get_user_model
from django.test import TestCase
from datetime import time
from rest_framework import status

User = get_user_model()

from user.serializers import jwt_token_of

from user.tests import UserFactory
from seminar.models import Seminar, UserSeminar


class SeminarFactory(DjangoModelFactory):
    class Meta:
        model = Seminar

    @classmethod
    def create(cls, **kwargs):
        user = Seminar.objects.create(**kwargs)
        return user


class PostSeminarTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.participant = UserFactory(
            username='p1',
            password='p1',
            first_name='pone',
            last_name='pone',
            email='p1@snu.ac.kr',
            is_participant=True
        )
        cls.participant.participant.university = 'Seoul National University'
        cls.participant.participant.save()
        cls.participant_token = 'JWT ' + jwt_token_of(User.objects.get(email='p1@snu.ac.kr'))

        cls.instructor = UserFactory(
            username='i1',
            password='i1',
            first_name='ione',
            last_name='ione',
            email='i1@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor.instructor.company = 'Naver'
        cls.instructor.instructor.year = 2
        cls.instructor.instructor.save()
        cls.instructor_token = 'JWT ' + jwt_token_of(User.objects.get(email='i1@snu.ac.kr'))

        cls.post_data = {
            'name': 'Django',
            'capacity': 40,
            'count': 5,
            'time': '10:00'
        }

    def test_post_seminar_incomplete_request(self):
        # No Token
        response = self.client.post('/api/v1/seminar/',
                                    content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        # Not Instructor
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.participant_token)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        # Wrong name, capacity, count, time
        self.post_data.pop('name')
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.post_data['name'] = ''
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.post_data['name'] = 'Django'

        self.post_data.pop('capacity')
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.post_data['capacity'] = 'string'
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.post_data['capacity'] = -1
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.post_data['capacity'] = 40

        self.post_data.pop('count')
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.post_data['count'] = 'string'
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.post_data['count'] = -1
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.post_data['count'] = 5

        self.post_data.pop('time')
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        self.post_data['time'] = '10-00'
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.post_data['time'] = '10:00'

    def test_post_seminar(self):
        # Without online data
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        data = response.json()
        self.assertIn('id', data)
        self.assertEqual(data['name'], 'Django')
        self.assertEqual(data['capacity'], 40)
        self.assertEqual(data['count'], 5)
        self.assertEqual(data['time'], '10:00:00')
        self.assertEqual(data['online'], True)
        self.assertIsNotNone(data['instructors'])
        self.assertListEqual(data['participants'], [])

        instructor = data['instructors'][0]
        self.assertIn('id', instructor)
        self.assertEqual(instructor['username'], 'i1')
        self.assertEqual(instructor['email'], 'i1@snu.ac.kr')
        self.assertEqual(instructor['first_name'], 'ione')
        self.assertEqual(instructor['last_name'], 'ione')
        self.assertIn('joined_at', instructor)

        self.assertEqual(Seminar.objects.count(), 1)
        self.assertEqual(User.objects.get(email="i1@snu.ac.kr").user_seminars.filter(role="instructor").count(), 1)

        # With online data
        self.post_data['online'] = False
        response = self.client.post('/api/v1/seminar/',
                                    data=self.post_data,
                                    content_type='application/json',
                                    HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        data = response.json()
        self.assertEqual(data['online'], False)


class PutSeminarSeminarIdTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.participant = UserFactory(
            username='p1',
            password='p1',
            first_name='pone',
            last_name='pone',
            email='p1@snu.ac.kr',
            is_participant=True
        )
        cls.participant.participant.university = 'Seoul National University'
        cls.participant.participant.save()
        cls.participant_token = 'JWT ' + jwt_token_of(User.objects.get(email='p1@snu.ac.kr'))

        cls.participant2 = UserFactory(
            username='p2',
            password='p2',
            first_name='ptwo',
            last_name='ptwo',
            email='p2@snu.ac.kr',
            is_participant=True
        )
        cls.participant2.participant.university = 'Stanford University'
        cls.participant2.participant.save()
        cls.participant2_token = 'JWT ' + jwt_token_of(User.objects.get(email='p2@snu.ac.kr'))

        cls.instructor = UserFactory(
            username='i1',
            password='i1',
            first_name='ione',
            last_name='ione',
            email='i1@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor.instructor.company = 'Naver'
        cls.instructor.instructor.year = 2
        cls.instructor.instructor.save()
        cls.instructor_token = 'JWT ' + jwt_token_of(User.objects.get(email='i1@snu.ac.kr'))

        cls.instructor2 = UserFactory(
            username='i2',
            password='i2',
            first_name='itwo',
            last_name='itwo',
            email='i2@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor2.instructor.company = 'Amazon'
        cls.instructor2.instructor.year = 3
        cls.instructor2.instructor.save()
        cls.instructor2_token = 'JWT ' + jwt_token_of(User.objects.get(email='i2@snu.ac.kr'))

        cls.seminar = SeminarFactory(
            name='React',
            capacity=40,
            count=5,
            time='16:30',
            online=True
        )
        cls.seminar_id = cls.seminar.id

        UserSeminar.objects.create(
            user=cls.participant,
            seminar=cls.seminar,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.participant2,
            seminar=cls.seminar,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.instructor,
            seminar=cls.seminar,
            role='instructor'
        )

    def test_put_seminar_seminarid_incomplete_request(self):
        # No Token
        response = self.client.put('/api/v1/seminar/1/',
                                   content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        # Not instructor
        response = self.client.put('/api/v1/seminar/%d/'%self.seminar_id,
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant_token)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        # Not instructor of this seminar
        response = self.client.put('/api/v1/seminar/%d/'%self.seminar_id,
                                   data={'capacity': 1},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor2_token)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        # No seminar with such seminar_id
        response = self.client.put('/api/v1/seminar/0/',
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        # Wrong capacity, count
        response = self.client.put('/api/v1/seminar/%d/'%self.seminar_id,
                                   data={'capacity': 1},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        response = self.client.put('/api/v1/seminar/%d/'%self.seminar_id,
                                   data={'capacity': 'string'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        response = self.client.put('/api/v1/seminar/%d/'%self.seminar_id,
                                   data={'count': 'string'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_put_seminar_seminarid(self):
        response = self.client.put('/api/v1/seminar/%d/'%self.seminar_id,
                                   data={},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        response = self.client.put('/api/v1/seminar/%d/'%self.seminar_id,
                                   data={'name': 'iOS',
                                         'capacity': 30,
                                         'count': 3,
                                         'time': '16:00',
                                         'online': False},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor_token)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()

        self.assertEqual(data['name'], 'iOS')
        self.assertEqual(data['capacity'], 30)
        self.assertEqual(data['count'], 3)
        self.assertEqual(data['time'], '16:00:00')
        self.assertEqual(data['online'], False)

        self.assertEqual(Seminar.objects.get(id=self.seminar_id).name, 'iOS')
        self.assertEqual(Seminar.objects.get(id=self.seminar_id).capacity, 30)
        self.assertEqual(Seminar.objects.get(id=self.seminar_id).count, 3)
        self.assertEqual(Seminar.objects.get(id=self.seminar_id).time, time(16, 0))
        self.assertEqual(Seminar.objects.get(id=self.seminar_id).online, False)

class GetSeminarTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.participant1 = UserFactory(
            username='p1',
            password='p1',
            first_name='pone',
            last_name='pone',
            email='p1@snu.ac.kr',
            is_participant=True
        )
        cls.participant1.participant.university = 'Seoul National University'
        cls.participant1.participant.save()
        cls.participant1_token = 'JWT ' + jwt_token_of(User.objects.get(email='p1@snu.ac.kr'))

        cls.participant2 = UserFactory(
            username='p2',
            password='p2',
            first_name='ptwo',
            last_name='ptwo',
            email='p2@snu.ac.kr',
            is_participant=True
        )
        cls.participant2.participant.university = 'Stanford University'
        cls.participant2.participant.save()
        cls.participant2_token = 'JWT ' + jwt_token_of(User.objects.get(email='p2@snu.ac.kr'))

        cls.instructor1 = UserFactory(
            username='i1',
            password='i1',
            first_name='ione',
            last_name='ione',
            email='i1@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor1.instructor.company = 'Naver'
        cls.instructor1.instructor.year = 2
        cls.instructor1.instructor.save()
        cls.instructor1_token = 'JWT ' + jwt_token_of(User.objects.get(email='i1@snu.ac.kr'))

        cls.instructor2 = UserFactory(
            username='i2',
            password='i2',
            first_name='itwo',
            last_name='itwo',
            email='i2@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor2.instructor.company = 'Amazon'
        cls.instructor2.instructor.year = 3
        cls.instructor2.instructor.save()
        cls.instructor2_token = 'JWT ' + jwt_token_of(User.objects.get(email='i2@snu.ac.kr'))

        cls.seminar1 = SeminarFactory(
            name='Django',
            capacity=40,
            count=5,
            time='10:00',
            online=True
        )
        cls.seminar1_id = cls.seminar1.id

        cls.seminar2 = SeminarFactory(
            name='Android',
            capacity=40,
            count=5,
            time='16:00',
            online=True
        )
        cls.seminar2_id = cls.seminar2.id

        UserSeminar.objects.create(
            user=cls.participant1,
            seminar=cls.seminar1,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.participant2,
            seminar=cls.seminar1,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.participant1,
            seminar=cls.seminar2,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.participant2,
            seminar=cls.seminar2,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.instructor1,
            seminar=cls.seminar1,
            role='instructor'
        )

        UserSeminar.objects.create(
            user=cls.instructor2,
            seminar=cls.seminar2,
            role='instructor'
        )

    def test_get_seminar_incomplete_request(self):
        # No Token
        response = self.client.get('/api/v1/seminar/',
                                   content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        # No seminar with such seminar_id
        response = self.client.get('/api/v1/seminar/0/',
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_get_seminar(self):
        # GET /api/v1/seminar/{seminar_id}/
        response = self.client.get('/api/v1/seminar/%d/'%self.seminar1_id,
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # GET /api/v1/seminar/ all
        response = self.client.get('/api/v1/seminar/',
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertEqual(len(data), 2)

        dataFirst = data[0]
        self.assertEqual(dataFirst['name'], 'Django')
        self.assertIn('instructors', dataFirst)
        self.assertEqual(dataFirst['participant_count'], 2)

        instructor = dataFirst['instructors'][0]
        self.assertIn('id', instructor)
        self.assertEqual(instructor['username'], 'i1')
        self.assertEqual(instructor['email'], 'i1@snu.ac.kr')
        self.assertEqual(instructor['first_name'], 'ione')
        self.assertEqual(instructor['last_name'], 'ione')
        self.assertIn('joined_at', instructor)

        # GET /api/v1/seminar/ empty
        response = self.client.get('/api/v1/seminar/?name=s',
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertEqual(data, [])

        # GET /api/v1/seminar/ reversed
        response = self.client.get('/api/v1/seminar/?name=o&order=earliest',
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertEqual(len(data), 2)

        dataFirst = data[0]
        self.assertEqual(dataFirst['name'], 'Android')


class PostSeminarSeminarIdUserTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.participant1 = UserFactory(
            username='p1',
            password='p1',
            first_name='pone',
            last_name='pone',
            email='p1@snu.ac.kr',
            is_participant=True
        )
        cls.participant1.participant.university = 'Seoul National University'
        cls.participant1.participant.save()
        cls.participant1_token = 'JWT ' + jwt_token_of(User.objects.get(email='p1@snu.ac.kr'))

        cls.participant2 = UserFactory(
            username='p2',
            password='p2',
            first_name='ptwo',
            last_name='ptwo',
            email='p2@snu.ac.kr',
            is_participant=True
        )
        cls.participant2.participant.university = 'Stanford University'
        cls.participant2.participant.accepted = False
        cls.participant2.participant.save()
        cls.participant2_token = 'JWT ' + jwt_token_of(User.objects.get(email='p2@snu.ac.kr'))

        cls.participant3 = UserFactory(
            username='p3',
            password='p3',
            first_name='pthree',
            last_name='pthree',
            email='p3@snu.ac.kr',
            is_participant=True
        )
        cls.participant3.participant.university = 'Carnegie Mellon University'
        cls.participant3.participant.save()
        cls.participant3_token = 'JWT ' + jwt_token_of(User.objects.get(email='p3@snu.ac.kr'))

        cls.participant4 = UserFactory(
            username='p4',
            password='p4',
            first_name='pfour',
            last_name='pfour',
            email='p4@snu.ac.kr',
            is_participant=True
        )
        cls.participant4.participant.university = 'Carnegie Mellon University'
        cls.participant4.participant.save()
        cls.participant4_token = 'JWT ' + jwt_token_of(User.objects.get(email='p4@snu.ac.kr'))

        cls.instructor1 = UserFactory(
            username='i1',
            password='i1',
            first_name='ione',
            last_name='ione',
            email='i1@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor1.instructor.company = 'Naver'
        cls.instructor1.instructor.year = 2
        cls.instructor1.instructor.save()
        cls.instructor1_token = 'JWT ' + jwt_token_of(User.objects.get(email='i1@snu.ac.kr'))

        cls.instructor2 = UserFactory(
            username='i2',
            password='i2',
            first_name='itwo',
            last_name='itwo',
            email='i2@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor2.instructor.company = 'Amazon'
        cls.instructor2.instructor.year = 3
        cls.instructor2.instructor.save()
        cls.instructor2_token = 'JWT ' + jwt_token_of(User.objects.get(email='i2@snu.ac.kr'))

        cls.instructor3 = UserFactory(
            username='i3',
            password='i3',
            first_name='ithree',
            last_name='ithree',
            email='i3@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor3.instructor.company = 'Google'
        cls.instructor3.instructor.year = 1
        cls.instructor3.instructor.save()
        cls.instructor3_token = 'JWT ' + jwt_token_of(User.objects.get(email='i3@snu.ac.kr'))

        cls.seminar1 = SeminarFactory(
            name='Django',
            capacity=2,
            count=5,
            time='10:00',
            online=True
        )
        cls.seminar1_id = cls.seminar1.id

        cls.seminar2 = SeminarFactory(
            name='Android',
            capacity=40,
            count=5,
            time='16:00',
            online=True
        )
        cls.seminar2_id = cls.seminar2.id

        UserSeminar.objects.create(
            user=cls.participant3,
            seminar=cls.seminar1,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.participant4,
            seminar=cls.seminar1,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.instructor1,
            seminar=cls.seminar1,
            role='instructor'
        )

        UserSeminar.objects.create(
            user=cls.instructor2,
            seminar=cls.seminar2,
            role='instructor'
        )

    def test_post_seminar_incomplete_request(self):
        # No Token
        response = self.client.get('/api/v1/seminar/1/user/',
                                   content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        # No seminar with such seminar_id
        response = self.client.post('/api/v1/seminar/0/user/',
                                   data={'role': 'participant'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        # Not accepted participant
        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   data={'role': 'participant'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant2_token)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        # No role
        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   data={},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        # Wrong role
        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   data={'role': 'wrong_role'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   data={'role': 'instructor'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar2_id,
                                   data={'role': 'participant'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor1_token)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        # Already participating in this seminar
        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   data={'role': 'participant'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant3_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   data={'role': 'instructor'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor1_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        # Already working as instructor in other seminar
        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar2_id,
                                   data={'role': 'instructor'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor1_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        # Over capacity
        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   data={'role': 'participant'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_post_seminar(self):
        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar2_id,
                                   data={'role': 'participant'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        data = response.json()
        self.assertIn('id', data)
        self.assertEqual(data['name'], 'Android')
        self.assertEqual(data['capacity'], 40)
        self.assertEqual(data['count'], 5)
        self.assertEqual(data['time'], '16:00:00')
        self.assertEqual(data['online'], True)
        self.assertIn('instructors', data)
        self.assertIn('participants', data)

        instructor = data['instructors'][0]
        self.assertIn('id', instructor)
        self.assertEqual(instructor['username'], 'i2')
        self.assertEqual(instructor['email'], 'i2@snu.ac.kr')
        self.assertEqual(instructor['first_name'], 'itwo')
        self.assertEqual(instructor['last_name'], 'itwo')
        self.assertIn('joined_at', instructor)

        participant = data['participants'][0]
        self.assertIn('id', participant)
        self.assertEqual(participant['username'], 'p1')
        self.assertEqual(participant['email'], 'p1@snu.ac.kr')
        self.assertEqual(participant['first_name'], 'pone')
        self.assertEqual(participant['last_name'], 'pone')
        self.assertIn('joined_at', participant)
        self.assertEqual(participant['is_active'], True)
        self.assertIsNone(participant['dropped_at'])

        self.assertEqual(UserSeminar.objects.filter(seminar__id=self.seminar2_id, role='participant').count(), 1)

        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar2_id,
                                   data={'role': 'instructor'},
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor3_token)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        data = response.json()
        self.assertEqual(len(data['instructors']), 2)

        instructor = data['instructors'][1]
        self.assertIn('id', instructor)
        self.assertEqual(instructor['username'], 'i3')
        self.assertEqual(instructor['email'], 'i3@snu.ac.kr')
        self.assertEqual(instructor['first_name'], 'ithree')
        self.assertEqual(instructor['last_name'], 'ithree')
        self.assertIn('joined_at', instructor)

        self.assertEqual(UserSeminar.objects.filter(seminar__id=self.seminar2_id, role='instructor').count(), 2)


class DeleteSeminarSeminarIdUserTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.participant1 = UserFactory(
            username='p1',
            password='p1',
            first_name='pone',
            last_name='pone',
            email='p1@snu.ac.kr',
            is_participant=True
        )
        cls.participant1.participant.university = 'Seoul National University'
        cls.participant1.participant.save()
        cls.participant1_token = 'JWT ' + jwt_token_of(User.objects.get(email='p1@snu.ac.kr'))

        cls.participant2 = UserFactory(
            username='p2',
            password='p2',
            first_name='ptwo',
            last_name='ptwo',
            email='p2@snu.ac.kr',
            is_participant=True
        )
        cls.participant2.participant.university = 'Stanford University'
        cls.participant2.participant.save()
        cls.participant2_token = 'JWT ' + jwt_token_of(User.objects.get(email='p2@snu.ac.kr'))

        cls.participant3 = UserFactory(
            username='p3',
            password='p3',
            first_name='pthree',
            last_name='pthree',
            email='p3@snu.ac.kr',
            is_participant=True
        )
        cls.participant3.participant.university = 'Carnegie Mellon University'
        cls.participant3.participant.save()
        cls.participant3_token = 'JWT ' + jwt_token_of(User.objects.get(email='p3@snu.ac.kr'))

        cls.instructor1 = UserFactory(
            username='i1',
            password='i1',
            first_name='ione',
            last_name='ione',
            email='i1@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor1.instructor.company = 'Naver'
        cls.instructor1.instructor.year = 2
        cls.instructor1.instructor.save()
        cls.instructor1_token = 'JWT ' + jwt_token_of(User.objects.get(email='i1@snu.ac.kr'))

        cls.instructor2 = UserFactory(
            username='i2',
            password='i2',
            first_name='itwo',
            last_name='itwo',
            email='i2@snu.ac.kr',
            is_instructor=True
        )
        cls.instructor2.instructor.company = 'Amazon'
        cls.instructor2.instructor.year = 3
        cls.instructor2.instructor.save()
        cls.instructor2_token = 'JWT ' + jwt_token_of(User.objects.get(email='i2@snu.ac.kr'))

        cls.seminar1 = SeminarFactory(
            name='Django',
            capacity=40,
            count=5,
            time='10:00',
            online=True
        )
        cls.seminar1_id = cls.seminar1.id

        UserSeminar.objects.create(
            user=cls.participant1,
            seminar=cls.seminar1,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.participant2,
            seminar=cls.seminar1,
            role='participant'
        )

        UserSeminar.objects.create(
            user=cls.instructor1,
            seminar=cls.seminar1,
            role='instructor'
        )

    def test_delete_seminar_incomplete_request(self):
        # No Token
        response = self.client.delete('/api/v1/seminar/1/user/',
                                   content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        # No seminar with such seminar_id
        response = self.client.delete('/api/v1/seminar/0/user/',
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant1_token)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        # User is instructor
        response = self.client.delete('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.instructor1_token)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_delete_seminar(self):
        # Not participating
        response = self.client.delete('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant3_token)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertEqual(len(data['participants']), 2)

        # Drop
        response = self.client.delete('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant2_token)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn('id', data)
        self.assertEqual(data['name'], 'Django')
        self.assertEqual(data['capacity'], 40)
        self.assertEqual(data['count'], 5)
        self.assertEqual(data['time'], '10:00:00')
        self.assertEqual(data['online'], True)
        self.assertIn('instructors', data)
        self.assertIn('participants', data)

        instructor = data['instructors'][0]
        self.assertIn('id', instructor)
        self.assertEqual(instructor['username'], 'i1')
        self.assertEqual(instructor['email'], 'i1@snu.ac.kr')
        self.assertEqual(instructor['first_name'], 'ione')
        self.assertEqual(instructor['last_name'], 'ione')
        self.assertIn('joined_at', instructor)

        participant = data['participants'][0]
        self.assertIn('id', participant)
        self.assertEqual(participant['username'], 'p1')
        self.assertEqual(participant['email'], 'p1@snu.ac.kr')
        self.assertEqual(participant['first_name'], 'pone')
        self.assertEqual(participant['last_name'], 'pone')
        self.assertIn('joined_at', participant)
        self.assertEqual(participant['is_active'], True)
        self.assertIsNone(participant['dropped_at'])

        participant = data['participants'][1]
        self.assertIn('id', participant)
        self.assertEqual(participant['username'], 'p2')
        self.assertEqual(participant['email'], 'p2@snu.ac.kr')
        self.assertEqual(participant['first_name'], 'ptwo')
        self.assertEqual(participant['last_name'], 'ptwo')
        self.assertIn('joined_at', participant)
        self.assertEqual(participant['is_active'], False)
        self.assertIsNotNone(participant['dropped_at'])

        self.assertIsNotNone(UserSeminar.objects.get(user__id=self.participant2.id, seminar__id=self.seminar1_id).dropped_at)

        # Can't re-enter the seminar
        response = self.client.post('/api/v1/seminar/%d/user/'%self.seminar1_id,
                                   content_type='application/json',
                                   HTTP_AUTHORIZATION=self.participant2_token)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
