from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

def addPath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

thisDir = os.path.dirname(__file__)
libPath = os.path.join(thisDir, 'lib')
addPath(libPath)