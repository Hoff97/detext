from datetime import datetime

from django.test import TestCase

from detext.server.models import MathSymbol


class MathSymbolTestCase(TestCase):
    def setUp(self):
        self.o = MathSymbol.objects.create(name = 'Test', description = 'Test', timestamp = datetime.now())

    def tearDown(self):
        self.o.delete()

    def test_can_get_math_symbol(self):
        g = MathSymbol.objects.get(name = 'Test')
        self.assertEqual(g.name, self.o.name)
