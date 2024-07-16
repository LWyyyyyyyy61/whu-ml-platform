from django.test import TestCase
import unittest
from django.test import Client
from django.contrib.auth.models import User
import os
import unittest
from django.test import Client, TestCase
from django.urls import reverse
from frontend.machinelearn.models import rback, histy

class LoginTestCase(unittest.TestCase):
    def setUp(self):
        self.client = Client()
        self.username = 'testuser'
        self.password = 'testpassword'
        self.user = User.objects.create_user(username=self.username, password=self.password)

    def test_login_success(self):
        response = self.client.post('/login/', {'user': self.username, 'pwd': self.password})
        self.assertEqual(response.status_code, 302)  # Check if the response is a redirect
        self.assertEqual(response.url, '/home')  # Check if the redirect URL is correct

    def test_login_failure(self):
        response = self.client.post('/login/', {'user': self.username, 'pwd': 'wrongpassword'})
        self.assertEqual(response.status_code, 200)  # Check if the response is a success
        self.assertContains(response, '用户名或密码错误')  # Check if the error message is displayed

    def tearDown(self):
        self.user.delete()


class RegisterTestCase(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_register_success(self):
        response = self.client.post('/register/', {
            'user': 'testuser',
            'pwd': 'testpassword',
            'pwd_confirm': 'testpassword',
            'e-mail': 'test@example.com',
        })
        self.assertEqual(response.status_code, 302)  # Check if the registration was successful
        self.assertTrue(User.objects.filter(username='testuser').exists())  # Check if the user was created

    def test_register_password_mismatch(self):
        response = self.client.post('/register/', {
            'user': 'testuser',
            'pwd': 'testpassword',
            'pwd_confirm': 'mismatchpassword',
            'e-mail': 'test@example.com',
        })
        self.assertEqual(response.status_code, 200)  # Check if the registration failed
        self.assertFalse(User.objects.filter(username='testuser').exists())  # Check if the user was not created

    def test_register_existing_username(self):
        User.objects.create_user(username='existinguser', password='testpassword')
        response = self.client.post('/register/', {
            'user': 'existinguser',
            'pwd': 'testpassword',
            'pwd_confirm': 'testpassword',
            'e-mail': 'test@example.com',
        })
        self.assertEqual(response.status_code, 200)  # Check if the registration failed
        self.assertFalse(User.objects.filter(email='test@example.com').exists())  # Check if the user was not created

class UploadFileTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('upload_file')

    def test_upload_file_success(self):
        # Create a test file
        file = open('test_file.csv', 'w')
        file.write('test data')
        file.close()

        # Create a POST request with the file
        with open('test_file.csv', 'rb') as f:
            response = self.client.post(self.url, {'file': f})

        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)

        # Assert that the file was saved and processed correctly
        self.assertTrue(rback.objects.exists())
        self.assertTrue(histy.objects.exists())

    def test_upload_file_invalid_form(self):
        # Create a POST request without a file
        response = self.client.post(self.url)

        # Assert that the response is not successful
        self.assertNotEqual(response.status_code, 200)

        # Assert that the file was not saved
        self.assertFalse(rback.objects.exists())
        self.assertFalse(histy.objects.exists())

    def tearDown(self):
        # Delete the test file
        os.remove('test_file.csv')

if __name__ == '__main__':
    unittest.main()