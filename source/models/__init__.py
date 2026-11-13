"""Discover concrete model classes; no hand-maintained model registry."""
import importlib
import inspect
import pkgutil
import re
from .base import ForecastModel


def discover_models() -> dict[str, type[ForecastModel]]:
    found = {}
    for entry in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
        if entry.name.startswith('_') or entry.name == 'base':
            continue
        module = importlib.import_module(f'{__name__}.{entry.name}')
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or not issubclass(cls, ForecastModel) or inspect.isabstract(cls):
                continue
            name = getattr(cls, 'name', '')
            if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
                raise ValueError(f'{cls.__name__}: invalid model name {name!r}')
            if name in found:
                raise ValueError(f'Duplicate model name: {name}')
            if not {'seed', 'threads'}.issubset(inspect.signature(cls).parameters):
                raise TypeError(f'{name}: constructor must accept seed and threads')
            variants = cls.variants()
            if not variants:
                raise ValueError(f'{name}: declare at least one variant')
            for variant, spec in variants.items():
                if not re.fullmatch(r'[a-z][a-z0-9_]*', variant):
                    raise ValueError(f'{name}: invalid variant name {variant!r}')
                budgets = spec['budgets']
                if not budgets or budgets != sorted(set(budgets)) or any(type(b) is not int or b < 1 for b in budgets):
                    raise ValueError(f'{name}/{variant}: budgets must be increasing positive integers')
                if type(spec['training_cap']) is not int or spec['training_cap'] < max(budgets):
                    raise ValueError(f'{name}/{variant}: training capacity is too small')
                if not isinstance(spec['parameters'], dict):
                    raise TypeError(f'{name}/{variant}: parameters must be a dictionary')
            found[name] = cls
    if not found:
        raise ValueError('No concrete ForecastModel classes discovered')
    return found
