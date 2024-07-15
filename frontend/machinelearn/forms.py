from django import forms
from .models import rback,histy
class UploadFileForm(forms.Form):
    file = forms.FileField()
    target_column = forms.CharField(max_length=100)
    
class returnbac(forms.ModelForm):
    class Meta:
        model=rback
        fields=['niming','content']
        widgets = {
            'niming': forms.Select(attrs={'class': 'form-control','style':"border-color:white;background-color: transparent;"}),
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 5, 'placeholder': '输入内容','style':"border-color:white;background-color: transparent;"}),
        }
class his(forms.ModelForm):
    class Meta:
        model=histy
        fields=['usecontent']
        widgets={
            'usecontent':forms.Textarea(attrs={'class': 'form-control', 'rows': 5, 'placeholder': '记录你的工作','style':"border-color:white;background-color: transparent;"})
        }