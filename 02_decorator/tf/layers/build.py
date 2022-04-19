from decorator.decorator import ClsDecorator

_my_decorator = ClsDecorator()

def get_layer(name, cfg):

    return _my_decorator.get(name)(cfg)