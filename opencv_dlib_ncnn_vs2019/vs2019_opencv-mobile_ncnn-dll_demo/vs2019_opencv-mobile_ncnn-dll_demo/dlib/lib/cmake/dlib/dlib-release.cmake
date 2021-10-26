#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dlib::dlib" for configuration "Release"
set_property(TARGET dlib::dlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(dlib::dlib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/dlib19.20.0_release_64bit_msvc1928.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS dlib::dlib )
list(APPEND _IMPORT_CHECK_FILES_FOR_dlib::dlib "${_IMPORT_PREFIX}/lib/dlib19.20.0_release_64bit_msvc1928.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
