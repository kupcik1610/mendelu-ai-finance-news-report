"""
URL routes for the research application.
"""

from django.urls import path
from . import views

app_name = 'research'

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    path('search/', views.search, name='search'),
    path('results/<int:pk>/', views.results, name='results'),
    path('status/<int:pk>/', views.status, name='status'),

    # History
    path('history/', views.history, name='history'),

    # API endpoints
    path('api/research/<int:pk>/', views.api_research, name='api_research'),
]
