# Description:
#   Build rule for Python and Numpy.

cc_library(
    name = "python",
    hdrs = glob([
        "python@2/2.7.15/Frameworks/Python.framework/Versions/2.7/include/python2.7/*.h",
        "numpy/1.15.2/lib/python2.7/site-packages/numpy/core/include/numpy/*.h",
    ]),
    includes = [
        "python@2/2.7.15/Frameworks/Python.framework/Versions/2.7/include/python2.7",
        "numpy/1.15.2/lib/python2.7/site-packages/numpy/core/include",
    ],
    visibility = ["//visibility:public"],
)
