from unittest import TestCase

from tensor_dynamic.lazyprop import lazyprop, clear_all_lazyprops


class TestLazyprop(TestCase):

    def test_clear_all(self):
        class PropClass():
            STATIC_VAL = None

            @lazyprop
            def lazyprop(self):
                return self.STATIC_VAL

        prop_class = PropClass()
        prop_class.STATIC_VAL = 1
        self.assertEquals(prop_class.lazyprop, 1)

        prop_class.STATIC_VAL = 2
        self.assertEquals(prop_class.lazyprop, 1)

        clear_all_lazyprops(prop_class)
        self.assertEquals(prop_class.lazyprop, 2)