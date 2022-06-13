from pprint import pprint

# some functions are desirable in almost every object, so it's useful to have a
# BaseObject class that implements them


class BaseObject(object):
    def print(self) -> None:
        pprint(vars(self))
