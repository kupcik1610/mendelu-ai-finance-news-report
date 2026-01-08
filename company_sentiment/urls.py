"""
URL configuration for company_sentiment project.
"""
from django.urls import path, include

urlpatterns = [
    path('', include('research.urls')),
]
