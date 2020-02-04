# Copyright (c) 2019 Eleftherios Avramidis <el.avramidis@gmail.com>
#
# Distributed under The MIT License (MIT)
# See accompanying file LICENSE.txt
#---------------------------------------------------------------------------//

FIND_PACKAGE(Python3 COMPONENTS Development NumPy)
if (${Python3_FOUND})
    message(STATUS "PYTHON3 FOUND")
    message(STATUS "Python3 include dir: " ${Python3_INCLUDE_DIRS})
    message(STATUS "Python3 libraries: " ${Python3_LIBRARIES})
    message(STATUS "Python3 NumPy include dir: " ${Python3_NumPy_INCLUDE_DIRS})
else ()
    message(FATAL_ERROR "PYTHON3 NOT FOUND")
endif ()

include(FetchContent)
FetchContent_Declare(Chronos
        GIT_REPOSITORY http://github.com/avramidis/chronos.git
        GIT_TAG origin/master)
FetchContent_MakeAvailable(Chronos)

include(FetchContent)
FetchContent_Declare(cxxplot
        GIT_REPOSITORY http://github.com/avramidis/cxxplot.git
        GIT_TAG origin/master)
FetchContent_MakeAvailable(cxxplot)