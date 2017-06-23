from unittest import TestCase

from tensor_dynamic.lazyprop import lazyprop, clear_all_lazyprops, subscribe_to_lazy_prop, unsubscribe_from_lazy_prop, \
    clear_lazyprop_on_lazyprop_cleared, clear_lazyprop


class _PropClass(object):
    STATIC_VAL = None

    @lazyprop
    def lazyprop(self):
        return self.STATIC_VAL


class TestLazyprop(TestCase):
    def test_clear_all(self):
        prop_class = _PropClass()
        prop_class.STATIC_VAL = 1
        self.assertEquals(prop_class.lazyprop, 1)

        prop_class.STATIC_VAL = 2
        self.assertEquals(prop_class.lazyprop, 1)

        clear_all_lazyprops(prop_class)
        self.assertEquals(prop_class.lazyprop, 2)

    def test_subscribe_lazy_prop_change(self):
        prop_class = _PropClass()
        checker = []
        subscribe_to_lazy_prop(prop_class, 'lazyprop',
                               lambda _: checker.append(1))

        clear_all_lazyprops(prop_class)

        self.assertEqual(checker, [1])

    def test_unsubscribe_lazy_prop_change(self):
        prop_class = _PropClass()
        checker = []
        func = lambda _: checker.append(1)
        subscribe_to_lazy_prop(prop_class, 'lazyprop', func)

        clear_all_lazyprops(prop_class)

        self.assertEqual(len(checker), 1)

        unsubscribe_from_lazy_prop(prop_class, 'lazyprop', func)

        clear_all_lazyprops(prop_class)

        self.assertEqual(len(checker), 1)

    def test_clear_lazyprop_on_lazyprop_cleared(self):
        prop_class_1 = _PropClass()
        prop_class_2 = _PropClass()

        clear_lazyprop_on_lazyprop_cleared(prop_class_2, 'lazyprop',
                                           prop_class_1, 'lazyprop')

        prop_class_1.STATIC_VAL = 1
        prop_class_2.STATIC_VAL = 2

        self.assertEqual(prop_class_1.lazyprop, 1)
        self.assertEqual(prop_class_2.lazyprop, 2)

        prop_class_1.STATIC_VAL = 3
        prop_class_2.STATIC_VAL = 4

        clear_lazyprop(prop_class_1, 'lazyprop')

        self.assertEqual(prop_class_1.lazyprop, 3)
        self.assertEqual(prop_class_2.lazyprop, 4)