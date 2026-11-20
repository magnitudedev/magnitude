"""mrig, the magnitude rig: questions to the hardware, answered by instruments on the hosts.

The `host` subpackage runs on the evaluated host itself (standard library, Python 3.9+);
everything else runs on the controlling machine.
"""

from pkgutil import extend_path

# The deployed host zip contains only `magnitude_rig.host` and sits first on
# PYTHONPATH for recorded remote jobs. Extending `__path__` lets `host` resolve
# from the controller's zip while allowing sibling modules (such as
# `magnitude_rig.store`) to resolve from the job's own installed package.
__path__ = extend_path(__path__, __name__)
