cc_library(
    name = "glib",
    srcs = ["lib/libglib-2.0.0.dylib"],
    hdrs = glob(["include/glib-2.0/**/*.h", "lib/glib-2.0/**/*.h"]),
    includes = ["include/glib-2.0", "lib/glib-2.0/include"],
    visibility = ["//visibility:public"],
    linkopts = ["-liconv"],
)
