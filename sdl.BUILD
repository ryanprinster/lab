# Description:
#   Build rule for SDL2.

cc_library(
    name = "sdl2",
    hdrs = glob(["include/SDL2/*.h"]),
    srcs = ["lib/libSDL2-2.0.0.dylib"],
    defines = ["_REENTRANT"],
    includes = ["include/SDL2"],
    visibility = ["//visibility:public"],
)
