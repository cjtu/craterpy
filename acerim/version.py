# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_maintenance = 5  # use '' for first of series, number for 1 and above
_version_extra = '' # dev for in development (NEVER push with dev), '' for full release

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_maintenance:
    _ver.append(_version_maintenance)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))
