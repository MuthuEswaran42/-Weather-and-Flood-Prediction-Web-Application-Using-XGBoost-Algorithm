# project/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # App URLs
    path('', include('WF.urls')),            # Main site (home, about, etc.)
    path('account/', include('account.urls')),  # Login, register, profile
    # Chat groups, rooms, etc.
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
