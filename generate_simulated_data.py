import clr, os, winreg
from itertools import islice

# This boilerplate requires the 'pythonnet' module.
# The following instructions are for installing the 'pythonnet' module via pip:
#    1. Ensure you are running a Python version compatible with PythonNET. Check the article "ZOS-API using Python.NET" or
#    "Getting started with Python" in our knowledge base for more details.
#    2. Install 'pythonnet' from pip via a command prompt (type 'cmd' from the start menu or press Windows + R and type 'cmd' then enter)
#
#        python -m pip install pythonnet

# determine the Zemax working directory
aKey = winreg.OpenKey(
    winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER),
    r"Software\Zemax",
    0,
    winreg.KEY_READ,
)
zemaxData = winreg.QueryValueEx(aKey, "ZemaxRoot")
NetHelper = os.path.join(
    os.sep, zemaxData[0], r"ZOS-API\Libraries\ZOSAPI_NetHelper.dll"
)
winreg.CloseKey(aKey)

# add the NetHelper DLL for locating the OpticStudio install folder
clr.AddReference(NetHelper)
import ZOSAPI_NetHelper

pathToInstall = ""
# uncomment the following line to use a specific instance of the ZOS-API assemblies
# pathToInstall = r'C:\C:\Program Files\Zemax OpticStudio'

# connect to OpticStudio
success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(pathToInstall)

zemaxDir = ""
if success:
    zemaxDir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
    print("Found OpticStudio at:   %s" + zemaxDir)
else:
    raise Exception("Cannot find OpticStudio")

# load the ZOS-API assemblies
clr.AddReference(os.path.join(os.sep, zemaxDir, r"ZOSAPI.dll"))
clr.AddReference(os.path.join(os.sep, zemaxDir, r"ZOSAPI_Interfaces.dll"))
import ZOSAPI

TheConnection = ZOSAPI.ZOSAPI_Connection()
if TheConnection is None:
    raise Exception("Unable to intialize NET connection to ZOSAPI")

TheApplication = TheConnection.ConnectAsExtension(0)
if TheApplication is None:
    raise Exception("Unable to acquire ZOSAPI application")

if TheApplication.IsValidLicenseForAPI == False:
    raise Exception(
        "License is not valid for ZOSAPI use.  Make sure you have enabled 'Programming > Interactive Extension' from the OpticStudio GUI."
    )

TheSystem = TheApplication.PrimarySystem
if TheSystem is None:
    raise Exception("Unable to acquire Primary system")


def reshape(data, x, y, transpose=False):
    """Converts a System.Double[,] to a 2D list for plotting or post processing

    Parameters
    ----------
    data      : System.Double[,] data directly from ZOS-API
    x         : x width of new 2D list [use var.GetLength(0) for dimension]
    y         : y width of new 2D list [use var.GetLength(1) for dimension]
    transpose : transposes data; needed for some multi-dimensional line series data

    Returns
    -------
    res       : 2D list; can be directly used with Matplotlib or converted to
                a numpy array using numpy.asarray(res)
    """
    if type(data) is not list:
        data = list(data)
    var_lst = [y] * x
    it = iter(data)
    res = [list(islice(it, i)) for i in var_lst]
    if transpose:
        return self.transpose(res)
    return res


def transpose(data):
    """Transposes a 2D list (Python3.x or greater).

    Useful for converting mutli-dimensional line series (i.e. FFT PSF)

    Parameters
    ----------
    data      : Python native list (if using System.Data[,] object reshape first)

    Returns
    -------
    res       : transposed 2D list
    """
    if type(data) is not list:
        data = list(data)
    return list(map(list, zip(*data)))


print("Connected to OpticStudio")

# The connection should now be ready to use.  For example:
print("Serial #: ", TheApplication.SerialCode)

# Insert Code Here

import numpy as np
from PIL import Image

TheNCE = TheSystem.NCE


def scale_image_vals(data: np.ndarray) -> np.ndarray:
    if data.max() <= 255:
        return data
    return 255 * data / data.max()


def get_image_from_detector(detector) -> Image:
    data = np.flipud(detector)
    processed = scale_image_vals(data)
    img = Image.fromarray(processed).convert("L")
    return img


def turn_entrance_mirror(x, y: float) -> None:
    TheNCE.GetObjectAt(5).TiltAboutX = x  # turn mirror
    TheNCE.GetObjectAt(5).TiltAboutY = y
    TheNCE.ReloadAllObjects()


# Set coordinate matrix
STEP = 0.1
x_tilt = np.arange(-5, 1.6, STEP)  # vertical tilt. Negative - looking down
y_tilt = np.arange(-3.5, 3, STEP)  # horizontal tilt. Negative - looking right


# x = 1.6, then y from 0 to 0
# x = -6, then y from -2 to 0

# y = -3.5, then x from 0 to 4.1
# y = 3, then x from -5 to 3

# Go through coordinate matrix
for x in x_tilt:
    for y in y_tilt:
        # Update x_tilt, y_tilt
        turn_entrance_mirror(x, y)

        # Call ray tracing
        NSCRayTrace = TheSystem.Tools.OpenNSCRayTrace()
        NSCRayTrace.ClearDetectors(7)
        NSCRayTrace.RunAndWaitForCompletion()
        NSCRayTrace.Close()

        # Get image from detector
        detector = TheNCE.GetAllDetectorDataSafe(7, 1)
        img = get_image_from_detector(detector)

        # Save image
        img.save(f"data/125x125_laser_x4_y6/x{x:.2f}_y{y:.2f}.jpg")
