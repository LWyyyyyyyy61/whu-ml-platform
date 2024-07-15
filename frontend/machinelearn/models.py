from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinLengthValidator, MaxLengthValidator, MinValueValidator, MaxValueValidator
# Create your models here.
class rback(models.Model):
    """反馈表"""
    niming_choices=(
        (1,"是"),
        (2,"否"),
    )
    username=models.CharField(verbose_name="用户名",max_length=64,blank=True,null=True)
    email=models.EmailField(verbose_name="邮箱",blank=True,null=True)
    niming=models.SmallIntegerField(verbose_name="是否匿名",choices=niming_choices)
    content=models.TextField(verbose_name="讨论或反馈内容")

class histy(models.Model):
    """历史记录"""
    username=models.CharField(verbose_name="用户名",max_length=64)
    usetime=models.DateTimeField(verbose_name="使用时间")
    usecontent=models.TextField(verbose_name="")