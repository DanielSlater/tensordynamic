_LAZYPROP_PREFIX = '_lazy_'


def lazyprop(fn):
    attr_name = _LAZYPROP_PREFIX + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def clear_all_lazyprops(self):
    for key in self.__dict__.keys():
        if key.startswith(_LAZYPROP_PREFIX):
            del self.__dict__[key]
