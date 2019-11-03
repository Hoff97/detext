from django.contrib import admin
from django.contrib.auth.models import User
from django.urls import include, path
from rest_framework import routers

from detext.server import views
from detext.server.models import ClassificationModel, MathSymbol, TrainImage

#admin.site.register(MathSymbol)
#admin.site.register(ClassificationModel)
#admin.site.register(TrainImage)

router = routers.DefaultRouter()
router.register(r'symbol', views.MathSymbolView)
router.register(r'model', views.ClassificationModelView)
router.register(r'image', views.TrainImageView)


# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
