import base64
import urllib

from django.conf import settings
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied, bad_request
from rest_framework.response import Response

from detext.server.models import MathSymbol
from detext.server.serializers import MathSymbolSerializer


class MathSymbolView(viewsets.ModelViewSet):
    queryset = MathSymbol.objects.all().order_by('timestamp')
    serializer_class = MathSymbolSerializer
    ordering = ['timestamp']

    """
    Destroy a model instance.
    """
    def destroy(self, request, *args, **kwargs):
        if request.user.id is None:
            raise PermissionDenied({
                "message": "Can only delete class symbol when logged in"
            })
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)

    """
    Create a model instance.
    """
    def create(self, request, *args, **kwargs):
        req_data = request.data.copy()

        if request.user.id is None:
            req_data['image'] = None

        code = req_data['latex']

        if 'image' not in req_data or req_data['image'] is None\
                or req_data['image'] == '':
            svg = self.get_latex_svg(code)
            req_data['image'] = svg

        serializer = self.get_serializer(data=req_data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED,
                        headers=headers)

    """
    Update a model instance.
    """
    def update(self, request, *args, **kwargs):
        if request.user.id is None:
            raise PermissionDenied({
                "message": "Can only update class symbol when logged in"
            })

        partial = kwargs.pop('partial', False)
        instance = self.get_object()

        if 'image' in request.data:
            err_msg = 'Image should not be contained in update request'
            return bad_request(request, Exception(err_msg))

        img = MathSymbol.objects.get(pk=instance.id).image
        instance.image = img

        serializer = self.get_serializer(instance, data=request.data,
                                         partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if instance.image is None or len(instance.image) == 0:
            instance.image = self.get_latex_svg(instance.latex)
            instance.save()

        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response('Ok')

    def get_latex_svg(self, code):
        url = settings.TEXSVG_URL + '?latex=' + code
        svg = urllib.request.urlopen(url).read()

        svg = base64.b64encode(svg).decode('utf-8')
        pre = 'data:image/svg+xml;base64,'
        svg = (pre + svg).encode('utf-8')
        return svg

    @action(detail=True, methods=['put'])
    def image(self, request, pk=None):
        if pk is None:
            return bad_request(request,
                               Exception('Primary key can not be null'))

        if 'image' not in request.data:
            err_msg = 'Image needs to be sent with this request'
            return bad_request(request, Exception(err_msg))

        if request.user.id is None:
            raise PermissionDenied({
                "message": "Can only update class image when logged in"
            })

        try:
            symbol = MathSymbol.objects.get(pk=pk)
            symbol.image = base64.b64decode(request.data['image'])
            symbol.save()

            return Response('Ok')
        except MathSymbol.DoesNotExist as e:
            return bad_request(request, e)
