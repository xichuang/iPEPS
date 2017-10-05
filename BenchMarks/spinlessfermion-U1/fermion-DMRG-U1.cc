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
    auto frac = input.getReal("frac");

    int N = Nx * Ny;
    int nfrac = N*frac;

    auto sites = Spinless(N, {"ConserveNf", true});

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
        ampo += VV, "N", bnd.s1, "N", bnd.s2;
    }
    for (auto i = 1; i <= Nx * Ny; ++i)
        ampo += -mu, "N", i;
    auto H = IQMPO(ampo);

    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        if (i % (N/nfrac) == 1)
            state.set(i, "Occ");
        else
            state.set(i, "Emp");
    }
    auto psi = IQMPS(state);
    printfln("Initial energy = %.5f", overlap(psi, H, psi));
    println("\nInitial Total QN", totalQN(psi));

    auto energy = dmrg(psi, H, sweeps, "Quiet");

    println("\nTotal QN of Ground State = ", totalQN(psi));
    std::cout<<"Dopping frac "<<frac<<", Occ number "<<nfrac<<" / " << N <<std::endl;
    std::cout << "mu = "<<mu << " E0 = " << energy << ", e0 per site = "<<energy/N << std::endl;
    groundE << mu << "  " << energy<< std::endl;

    groundE.close();
    return 0;
}

