import sys
import time
import pytest

try: # Check for local build
    import MDI_Library as mdi
except: # Check for installed package
    import mdi

with pytest.raises(Exception):
    mdi.MDI_Init("-role DRIVER -method TEST", None)
