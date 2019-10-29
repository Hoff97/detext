from django.contrib.auth.models import Group, User
from rest_framework import viewsets

from detext.server.models import MathSymbol
from detext.server.serializers import MathSymbolSerializer


class MathSymbolListView(viewsets.ReadOnlyModelViewSet):
    queryset = MathSymbol.objects.all()
    serializer_class = MathSymbolSerializer
    ordering = ['order']
