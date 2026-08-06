SET(TARGET "cute-fsi")

FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
SET(TARGET_SRC main.cc ${TARGET_SRC} ${TARGET_INC})

PROJECT(${TARGET} CXX)

CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

FIND_PACKAGE(deal.II 9.7.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

IF(deal.II_VERSION VERSION_LESS 9.8.0)
    ADD_COMPILE_DEFINITIONS(-DDEAL_II_LESS_9_8_0)
ENDIF()

IF (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/cute-fsi.prm)
    FILE(COPY ${PROJECT_SOURCE_DIR}/cute-fsi.prm DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
