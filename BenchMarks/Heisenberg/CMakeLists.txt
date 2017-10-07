set(progs Heisenberg-FFU Heisenberg-FU Heisenberg-SU)
foreach(prog ${progs})
    add_executable(${prog} "${prog}.cc")
    IF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
        target_link_libraries(${prog} iPEPS utilfunctions itensor-g
                mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread)
    ELSE()
        target_link_libraries(${prog} iPEPS utilfunctions itensor
                mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread)
    ENDIF()
endforeach()