_LAZYPROP_PREFIX = '__lazy__'


def lazyprop(fn):
    """
    A Python Property that will be evaluated once when first called. The result of this is cached and then returned on
    each subsequent call

    Parameters
    ----------
    fn : class method
        Will be made into a lazy prop

    Returns
    -------
    method as lazy prop

    """
    attr_name = _LAZYPROP_PREFIX + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def clear_all_lazyprops(self):
    """
    Clears all lazy prop from an object. This means they will be re-evaluated next time they are run

    Parameters
    ----------
    self : object
        The object we want to clear the lazy props from

    """
    for key in self.__dict__.keys():
        if key.startswith(_LAZYPROP_PREFIX):
            del self.__dict__[key]
