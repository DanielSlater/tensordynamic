from collections import defaultdict

_LAZY_PROP_VALUES = '__lazy_prop_values__'
_LAZY_PROP_SUBSCRIBERS = '__lazy_prop_subscribers__'


def lazyprop(fn):
    """A Python Property that will be evaluated once when first called. The result of this is cached and then returned on
    each subsequent call

    Examples:
        class A:
            @lazyprop
            def do_thing(self):
                return fib(2000)

        Because of lazyprop if you call do_thing twice the first time the value will be cached, then subsequent calls
        Will return the cached version saving having to compute fib(2000) again

    Args:
        fn (class method): Will be made into a lazy prop

    Returns:
        (method as lazy prop)
    """

    @property
    def _lazyprop(self):
        if not hasattr(self, _LAZY_PROP_VALUES):
            setattr(self, _LAZY_PROP_VALUES, {})
        lazy_props_dict = self.__dict__[_LAZY_PROP_VALUES]
        if fn.__name__ not in lazy_props_dict:
            lazy_props_dict[fn.__name__] = fn(self)
        return lazy_props_dict[fn.__name__]

    return _lazyprop


def subscribe_to_lazy_prop(object, property_name, on_change_func):
    assert isinstance(property_name, str)

    if not hasattr(object, _LAZY_PROP_SUBSCRIBERS):
        setattr(object, _LAZY_PROP_SUBSCRIBERS, defaultdict(lambda: set()))

    object.__dict__[_LAZY_PROP_SUBSCRIBERS][property_name].add(on_change_func)


def unsubscribe_from_lazy_prop(object, property_name, on_change_func):
    assert isinstance(property_name, str)

    if hasattr(object, _LAZY_PROP_SUBSCRIBERS):
        object.__dict__[_LAZY_PROP_SUBSCRIBERS][property_name].remove(on_change_func)


def clear_lazyprop(object, property_name):
    assert isinstance(property_name, str)

    if _LAZY_PROP_VALUES in object.__dict__:
        if property_name in object.__dict__[_LAZY_PROP_VALUES]:
            del object.__dict__[_LAZY_PROP_VALUES][property_name]

    if _LAZY_PROP_SUBSCRIBERS in object.__dict__:
        if property_name in object.__dict__[_LAZY_PROP_SUBSCRIBERS]:
            for fn in object.__dict__[_LAZY_PROP_SUBSCRIBERS][property_name]:
                fn(object)


def clear_all_lazyprops(object):
    """Clears all lazy prop from an object. This means they will be re-evaluated next time they are run

    Args:
        object (object): The object we want to clear the lazy props from
    """
    if _LAZY_PROP_VALUES in object.__dict__:
        del object.__dict__[_LAZY_PROP_VALUES]

    if _LAZY_PROP_SUBSCRIBERS in object.__dict__:
        for subscribers in object.__dict__[_LAZY_PROP_SUBSCRIBERS].values():
            for fn in subscribers:
                fn(object)


def clear_lazyprop_on_lazyprop_cleared(subscriber_object, subscriber_lazyprop,
                                       listen_to_object, listen_to_lazyprop=None):
    """Clear the lazyprop on the subscriber_object if the listen_to_object property is cleared

    Args:
        subscriber_object (object):
        subscriber_lazyprop (str):
        listen_to_object (object):
        listen_to_lazyprop (str):
    """
    if listen_to_lazyprop is None:
        listen_to_lazyprop = subscriber_lazyprop

    assert isinstance(listen_to_lazyprop, str)
    assert isinstance(subscriber_lazyprop, str)

    subscribe_to_lazy_prop(listen_to_object, listen_to_lazyprop,
                           lambda _: clear_lazyprop(subscriber_object, subscriber_lazyprop))
