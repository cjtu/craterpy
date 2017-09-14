""" 
Keeps track of the current acerim version. Versions are specified as a string
of the form "X.Y.Z" (major.minor.maintenance). Versions that are still in 
development append 'dev0' to the end (e.g., "X.Y.Zdev0").
"""
_version_major = 0
_version_minor = 1
_version_maintenance = ''  # use '' for each new major/minor release; int for 1+
_version_extra = '' # NEVER release with dev, use '' for full release


def concatenate_version(major, minor, maintenance, extra):
	""" Construct full version string to pass to setup.py """
	_ver = [major, minor]
	if maintenance:
	    _ver.append(maintenance)
	if extra:
	    _ver.append(extra)
	return '.'.join(map(str, _ver))

__version__ = concatenate_version(_version_major, _version_minor, 
									_version_maintenance, _version_extra)
