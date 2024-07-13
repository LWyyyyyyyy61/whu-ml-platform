from django import forms
from .models import rback
class UploadFileForm(forms.Form):
    file = forms.FileField()
    target_column = forms.CharField(max_length=100)
    
class returnbac(forms.ModelForm):
    class Meta:
        model=rback
        fields=['niming','content']
        widgets = {
            'niming': forms.Select(attrs={'class': 'form-control'}),
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 5, 'placeholder': 'Enter your feedback'}),
        }
