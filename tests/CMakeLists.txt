#set(progs iPEPS_construct iPEPS_Pre_Construct
#        SU SU_IQ SU_IQ_CB
#        CTM_construct ctmrg FU_IQ
#        SU_FU_IQ SU_FU_IQ_CB
#        SU_FFU_IQ SU_FFU_IQ_CB SU_FFU_FU_IQ_CB
#        isEqual QRdecomp svdcomp truncsvd randtest)
set(progs QRdecomp)
foreach(prog ${progs})
    add_executable(${prog} "${prog}.cc")
    target_link_libraries(${prog} iPEPS utilfunctions itensor
            mkl_intel_lp64 mkl_sequential mkl_core pthread)
#            mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread)
#    mkl_intel_lp64 mkl_sequential mkl_core pthread)
endforeach()