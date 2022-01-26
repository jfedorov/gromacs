find_package(VTune)

if (VTUNE_FOUND)
  message( STATUS "VTune found at " "${VTUNE_HOME}" )
  message( STATUS "VTune ITT lib found at " "${VTUNE_LIBRARY}" )
else ()
  message( FATAL_ERROR "VTune not found.")
  return()
endif()

include_directories(SYSTEM ${VTUNE_INCLUDE})

add_compile_definitions("ITT_INSTRUMENT" )
link_libraries( ${VTUNE_LIBRARY} )
