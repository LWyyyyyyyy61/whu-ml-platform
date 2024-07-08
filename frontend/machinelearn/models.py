from django.db import models
from django.core.validators import MinLengthValidator, MaxLengthValidator, MinValueValidator, MaxValueValidator
# Create your models here.
class User(models.Model):
    username=models.CharField(max_length=32)
    password=models.CharField(max_length=64)
    email=models.EmailField()
    REQUIRED_FIELDS = ['username','password']
    USERNAME_FIELD='username'
    phonenumber =models.CharField(max_length=11,default='1234567890')
    # models.DecimalField(
    #     max_digits=11,
    #     decimal_places=0,
    #     validators=[
    #         MinValueValidator(10000000000),
    #         MaxValueValidator(99999999999)
    #     ]
    # )
    
