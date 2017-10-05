//
// Created by xich on 17-3-2.
//

#include "itensor/all.h"
#include <iostream>
#include <fstream>

using namespace itensor;

int
main(int argc, char *argv[]) {
    if (argc < 2) {
        printfln("Usage: %s input_file", argv[0]);
        return 0;
    }
    auto input = InputGroup(argv[1], "input");
    auto Nx = input.getInt("Nx");
    auto Ny = input.getInt("Ny");
    auto yperiodic = input.getYesNo("yperiodic", false);
    auto lambda = input.getReal("lambda");
    auto VV = input.getReal("VV");
    auto mu = input.getReal("mu");
    auto dD = input.getReal("dD");

    int N = Nx * Ny;

    auto sites = Spinless(N, {"ConserveNf", false});

//    auto lattice = triangularLattice(Nx,Ny,{"YPeriodic=",yperiodic});
    //square lattice also available:
    auto lattice = squareLattice(Nx, Ny, {"YPeriodic=", yperiodic});

    auto sweeps = Sweeps(20);
    sweeps.maxm() = 10, 20, 100, 100, 200, 400;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 2;
    sweeps.noise() = 1E-7, 1E-8, 0.0;
    println(sweeps);

    std::fstream groundE;
    groundE.open("meanN.txt", std::ios::out);
    assert(groundE.is_open());

    auto ampo = AutoMPO(sites);
    for (auto bnd : lattice) {
        ampo += -lambda, "Cdag", bnd.s1, "C", bnd.s2;
        ampo += -lambda, "Cdag", bnd.s2, "C", bnd.s1;
        ampo += -dD, "Cdag", bnd.s1, "Cdag", bnd.s2;
        ampo += -dD, "C", bnd.s2, "C", bnd.s1;
        ampo += VV, "N", bnd.s1, "N", bnd.s2;
    }
    for (auto i = 1; i <= Nx * Ny; ++i)
        ampo += -mu, "N", i;
    auto H = IQMPO(ampo);
    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        if (i % 2 == 1)
            state.set(i, "Emp");
        else
            state.set(i, "Occ");
    }
    auto psi = IQMPS(state);
    printfln("Initial energy = %.5f", overlap(psi, H, psi));
    println("\nInitial Total QN", totalQN(psi));

    //
    // Begin the DMRG calculation
    //
    auto energyeven = dmrg(psi, H, sweeps, "Quiet");
    printfln("\nGround State Energy = %.10f", energyeven);
    printfln("\nGround State Energy per-site = %.10f", energyeven / (Nx * Ny));
    printfln("\nUsing overlap = %.10f", overlap(psi, H, psi));

    println("\nTotal QN of Ground State = ", totalQN(psi));

    state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        if (i % 2 == 1)
            state.set(i, "Emp");
        else
            state.set(i, "Occ");
    }
    state.set(N / 2 + 1, "Occ");
    psi = IQMPS(state);
    printfln("Initial energy = %.5f", overlap(psi, H, psi));
    println("\nInitial Total QN", totalQN(psi));

    auto energyodd = dmrg(psi, H, sweeps, "Quiet");
    for(auto i:range(N))
        Print(psi.A(i+1));

    Print(psi.A(1));

    auto lind=findtype(psi.A(1),Site);
    IQTensor S(lind),V,D;
    svd(psi.A(1),S,V,D);
    PrintDat(S);
    PrintDat(V);
    PrintDat(D);
    psi.position(3);
    Print(psi.A(3));

    printfln("\nGround State Energy = %.10f", energyodd);
    printfln("\nGround State Energy per-site = %.10f", energyodd / (Nx * Ny));
    printfln("\nUsing overlap = %.10f", overlap(psi, H, psi));
    println("\nTotal QN of Ground State = ", totalQN(psi));

    std::cout << dD << "  " << energyeven << "  " << energyodd << std::endl;
    groundE << dD << "  " << energyeven << "  " << energyodd << std::endl;


    groundE.close();
    return 0;
}

