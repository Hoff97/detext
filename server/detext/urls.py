from django.contrib import admin
from django.urls import include, path
from rest_framework import routers
from rest_framework.authtoken import views as auth_views

from detext.server.views.classification_model import ClassificationModelView
from detext.server.views.math_symbol import MathSymbolView
from detext.server.views.train_image import TrainImageView

router = routers.DefaultRouter()
router.register(r'symbol', MathSymbolView)
router.register(r'model', ClassificationModelView)
router.register(r'image', TrainImageView)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('api-auth/', include('rest_framework.urls',
                              namespace='rest_framework')),
    path('api/api-token-auth/', auth_views.obtain_auth_token)
]
