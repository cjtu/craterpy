"""Custom exceptions used by this project."""


class CraterAttributeError(AttributeError):
    pass


class CraterValueError(ValueError):
    pass


class LatLongOutOfBoundsError(ValueError):
    pass


class DataImportError(Exception):
    pass
