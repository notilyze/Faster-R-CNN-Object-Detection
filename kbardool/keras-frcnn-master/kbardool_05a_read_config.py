# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:13:18 2019

@author: paul
"""

import re, pickle, pprint, sys
from types import ModuleType
from collections.abc import Sequence, Mapping, Set
from contextlib import contextmanager


def pythonize(obj):
    if isinstance(obj, (str, bytes)):
        return obj
    if isinstance(obj, (Sequence, Set)):
        container = []
        for element in obj:
            container.append(pythonize(element))
        return container
    elif isinstance(obj, Mapping):
        container = {}
    else:
        container = {"$CLS": obj.__class__.__qualname__}
        if not hasattr(obj, "__dict__"):
            return repr(obj)
        obj = obj.__dict__
    for key, value in obj.items():
        container[key] = pythonize(value)
    return container


class FakeModule:
    def __getattr__(self, attr):
        cls = type(attr, (), {})
        setattr(self, attr, cls)
        return cls


def fake_importer(name, globals, locals, fromlist, level):
    module = sys.modules[name] = FakeModule()
    return module


@contextmanager
def fake_import_system():
    # With code lifted from https://github.com/jsbueno/extradict - MapGetter functionality
    builtins = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    original_import = builtins["__import__"]
    builtins["__import__"] = fake_importer
    yield None
    builtins["__import__"] = original_import


def unpickle_to_text(stream: bytes):
    # WARNING: this example will wreck havoc in loaded modules!
    # do not use as part of a complex system!!

    action_log = []

    with fake_import_system():
        result = pickle.loads(stream)

    pythonized = pythonize(result)

    return pprint.pformat(pythonized)


if __name__ == "__main__":
    print(unpickle_to_text(open(sys.argv[1], "rb").read()))