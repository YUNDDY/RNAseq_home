from django import forms

from uploads.core.models import Document


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('description', 'document', )

class HeatmapOptionsForm(forms.Form):
    option1 = forms.BooleanField(label='Option 1', required=False)
    option2 = forms.BooleanField(label='Option 2', required=False)
    option3 = forms.BooleanField(label='Option 3', required=False)