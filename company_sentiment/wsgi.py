"""
WSGI config for company_sentiment project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'company_sentiment.settings')

application = get_wsgi_application()
