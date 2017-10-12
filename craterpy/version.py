""" Keeps track of the current acerim version. Versions are specified as a
string of the form "X.Y.Z" (major.minor.maintenance). Versions still in
development append 'dev0' (e.g., "X.Y.Zdev0").
"""
_major = 0
_minor = 2
_maintenance = '2'  # use '' for new major/minor release; int for 1+
_extra = ''  # NEVER release with dev, use '' for full release


def concatenate_version(major, minor, maintenance, extra):
    """ Construct full version string to pass to setup.py """
    _ver = [major, minor]
    if maintenance:
        _ver.append(maintenance)
    if extra:
        _ver.append(extra)
    return '.'.join(map(str, _ver))


__version__ = concatenate_version(_major, _minor, _maintenance, _extra)
