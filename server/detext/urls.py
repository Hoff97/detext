from django.contrib import admin
from django.contrib.admin.sites import AlreadyRegistered
from django.contrib.auth.models import User
from django.urls import include, path
from rest_framework import routers
from rest_framework.authtoken import views as auth_views

from detext.server import views
from detext.server.models import ClassificationModel, MathSymbol, TrainImage

try:
    admin.site.register(MathSymbol)
except AlreadyRegistered:
    pass

try:
    admin.site.register(ClassificationModel)
except AlreadyRegistered:
    pass

try:
    admin.site.register(TrainImage)
except AlreadyRegistered:
    pass

router = routers.DefaultRouter()
router.register(r'symbol', views.MathSymbolView)
router.register(r'model', views.ClassificationModelView)
router.register(r'image', views.TrainImageView)


# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('api-token-auth/', auth_views.obtain_auth_token)
]
