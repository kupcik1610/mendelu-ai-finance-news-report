"""
Forms for the research application.
"""

from django import forms


class CompanySearchForm(forms.Form):
    """Form for searching/researching a company."""

    company_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g., Tesla, Apple, Microsoft',
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg '
                     'focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-lg',
            'autofocus': True,
        }),
        label='Company Name'
    )
