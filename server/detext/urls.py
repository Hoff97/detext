from django.contrib import admin
from django.urls import include, path
from rest_framework import routers
from rest_framework.authtoken import views as auth_views

from detext.server import views

router = routers.DefaultRouter()
router.register(r'symbol', views.MathSymbolView)
router.register(r'model', views.ClassificationModelView)
router.register(r'image', views.TrainImageView)


# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('api-auth/', include('rest_framework.urls',
                              namespace='rest_framework')),
    path('api/api-token-auth/', auth_views.obtain_auth_token)
]
