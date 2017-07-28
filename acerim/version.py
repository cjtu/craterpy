""" Keeps track of the current acerim version"""
# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_maintenance = 8  # use '' for first major/minor release, int for 1+
_version_extra = '' # NEVER release with dev0, use '' for full release

# Construct full version string to pass to setup.py
_ver = [_version_major, _version_minor]
if _version_maintenance:
    _ver.append(_version_maintenance)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))
