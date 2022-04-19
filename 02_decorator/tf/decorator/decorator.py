class ClsDecorator():
    def __init__(self):
        self.dic = {}

    def set(self):
        def deco_fn(_cls):
            name = _cls.__name__
            self.dic[name] = _cls
            return _cls

        return deco_fn

    def get(self, name):
        return self.dic.get(name)