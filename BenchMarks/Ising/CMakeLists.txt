set(progs Ising-SU Ising-FU Ising-FFU)
foreach(prog ${progs})
    add_executable(${prog} "${prog}.cc")
    target_link_libraries(${prog} iPEPS utilfunctions itensor
            mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread)

endforeach()