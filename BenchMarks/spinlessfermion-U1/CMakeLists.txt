set(progs fermion-DMRG-U1 fermion-SU-U1
        fermion-FFU-U1 fermion-FU-U1
        fermion-SU-CB-U1)
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