from django.contrib import admin

from detext.server.models import ClassificationModel, MathSymbol, TrainImage

admin.site.register(MathSymbol)
admin.site.register(ClassificationModel)
admin.site.register(TrainImage)
