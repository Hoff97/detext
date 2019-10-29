from django.contrib.auth.models import Group, User
from rest_framework import serializers

from detext.server.models import MathSymbol


class MathSymbolSerializer(serializers.ModelSerializer):
    class Meta:
        model = MathSymbol
        fields = ['name']
