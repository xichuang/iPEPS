//
// Created by xich on 16-11-14.
//

#include <iostream>
#include <fstream>

#include "iPEPS/iPEPS.h"
#include "iPEPS/CTM.h"

using IndexT = IQIndex;
using TensorT = ITensorT<IndexT>;
using iPEPS_TensorT  = iPEPS<IndexT>;
using CTM_TensorT  = CTM<IndexT>;

int
main(int argc, char *argv[]) {

    // Read simulateion parameters form "input_file"
    if (argc < 2) {
        printfln("Usage: %s input_file", argv[0]);
        return 0;
    }
    auto input = InputGroup(argv[1], "input");

    auto lambda = input.getReal("lambda");
    auto VV = input.getReal("VV");
    auto mu = input.getReal("mu");

    auto maxTimeStep = input.getInt("maxTimeStep");
    auto tstep = input.getReal("tstep");
    auto mindim = input.getInt("mindim");
    auto maxdim = input.getInt("maxdim");
    auto repeat = input.getInt("repeat");
    auto nrepeat = input.getInt("nrepeat");

    int bondD = 4;
    auto chi = bondD * 10;

    fstream mzout, iterout;
    mzout.open("meanN-FU.txt", ios::out);
    iterout.open("iter-FU.txt", ios::out);
    assert(mzout.is_open() && iterout.is_open());

    // set unitcell
    const int nx = 2;
    const int ny = 2;
    const int nsite = 4;
    auto sites = Spinless(nsite, {"ConserveNf", true});
    readFromFile("SU_site", sites);
    int cfg[nx * ny] = {0, 1, 2, 3};
    vector<int> uc(cfg, cfg + nx * ny);

    //define Hamitonian;
    auto hab = [&](const int i, const int j) {
        TensorT hh = sites.op("Adag", i + 1) * sites.op("A", j + 1);
        hh += sites.op("A", i + 1) * sites.op("Adag", j + 1);
        hh -= 0.25 * mu * sites.op("N", i + 1) * sites.op("Id", j + 1);
        hh -= 0.25 * mu * sites.op("Id", i + 1) * sites.op("N", j + 1);
        hh += VV * sites.op("N", i + 1) * sites.op("N", j + 1);

        return hh;
    };

    IndexT symInd;
    vector<TensorT> As;
    vector<IndexT> Links;

    readFromFile(format("SU_SymInd_%d_%d", nrepeat, bondD), symInd);
    for (auto i:range(nsite)) {
        TensorT Ai;
        readFromFile(format("SU_As_%d_%d_%d", nrepeat, bondD, i), Ai);
        As.push_back(Ai);
    }

    for (auto i:range(2 * nsite)) {
        IndexT Li;
        readFromFile(format("SU_Links_%d_%d_%d", nrepeat, bondD, i), Li);
        Links.push_back(Li);
    }

    CTM_TensorT ctm(uc, As, Links, symInd,
                    nsite, nx, ny, chi, {"Fermionic", true});
//            ctm.randEnv();
    auto dctm = ctm.CTMRG(40);

    cout << "SU results for (dim,chi) = ( " << bondD << ", " << chi << " ):" << endl;
    vector<Real> mz;
    for (auto i:range(nsite)) {
        auto line = i / nx;
        auto col = i % nx;
        mz.push_back(ctm.exp_site(sites.op("N", i + 1), i));
        cout << "N at (line,col) = ( " << line << " , " << col << " ) equals " << mz[i] << endl;
    }
    auto mztot = accumulate(mz.begin(), mz.end(), 0.0) / nsite;
    cout << "meanN = " << mztot << endl;

    vector<Real> energy;
    for (auto i:range(2 * nsite)) {
        auto abInd = calcABInd(i, nx, ny, uc);
        auto hh = hab(abInd[0], abInd[1]);

        energy.push_back(ctm.exp_bond(hh, i));
        cout << "Energy at bond " << i << " ) = " << energy[i] << endl;
    }
    auto energytot = accumulate(energy.begin(), energy.end(), 0.0) / nsite;
    cout << "Total energy = " << energytot << endl;
    Real snorm = ctm.stateNorm();
    cout << "Norm = " << snorm << endl;

    mzout << bondD << "  " << chi << "  " << snorm << "  " << 0 << "  "
          << mztot << "  " << energytot << "  " << 1.0 << "  " << dctm << endl;

    Real mzold = abs(mztot);
    Real energyold = energytot;

    Real dCTM = 0.0;
    Real dFU = 0.0;
    Real timeprop = 0.0;
    int ntimestep=0;
    for (ntimestep = 0; ntimestep < maxTimeStep; ++ntimestep) {
        cout<<"dim = "<<bondD<<", time step = "<<ntimestep<<endl;
        timeprop += tstep;
        dCTM = 0.0;
        dFU = 0.0;
        for (auto i:range(2 * nsite - 1)) {
            auto abInd = calcABInd(i, nx, ny, uc);
            auto hh = hab(abInd[0],abInd[1]);
            auto G = expHermitian(hh, -tstep / 2.0);
//            auto dFU0 = ctm.reduceUpdate(G, i);
            auto dFU0 = ctm.fastUpdate(G, i);
            auto dCTM0 = ctm.CTMRG();
            dCTM += dCTM0;
            dFU += dFU0;
        }
        {
            auto i = 2 * nsite - 1;
            auto abInd = calcABInd(i, nx, ny, uc);
            auto hh = hab(abInd[0],abInd[1]);
            auto G = expHermitian(hh, -tstep);
//            auto dFU0 = ctm.reduceUpdate(G, i);
            auto dFU0 = ctm.fastUpdate(G, i);
            auto dCTM0 = ctm.CTMRG();
            dCTM += dCTM0;
            dFU += dFU0;
        }
        for (auto j:range(2 * nsite - 1)) {
            auto i = 2 * nsite - j - 2;
            auto abInd = calcABInd(i, nx, ny, uc);
            auto hh = hab(abInd[0],abInd[1]);
            auto G = expHermitian(hh, -tstep / 2.0);
//            auto dFU0 = ctm.reduceUpdate(G, i);
            auto dFU0 = ctm.fastUpdate(G, i);
            auto dCTM0 = ctm.CTMRG();
            dCTM += dCTM0;
            dFU += dFU0;
        }

        if ((ntimestep + 1) % 100 == 0) {
            cout << "time step: " << ntimestep + 1 << endl;
            dCTM = ctm.CTMRG();

            mz.clear();
            mztot = 0.0;
            for (auto i:range(nsite)) {
                auto line = i / nx;
                auto col = i % nx;
                mz.push_back(ctm.exp_site(sites.op("N", i + 1), i));
            }
            mztot = accumulate(mz.begin(),mz.end(),0.0)/nsite;

            energy.clear();
            for (auto i:range(2 * nsite)) {
                auto abInd = calcABInd(i, nx, ny, uc);
                auto hh = hab(abInd[0],abInd[1]);

                energy.push_back(ctm.exp_bond(hh, i));
            }
            energytot = accumulate(energy.begin(), energy.end(), 0.0) / nsite;

            snorm = ctm.stateNorm();

            auto dmz = abs(mztot) - mzold;
            auto denergy = energytot - energyold;
            mzold = abs(mztot);
            energyold = energytot;

            for(auto i:range(nsite))
                writeToFile(format("FU_As_%d_%d_%d", bondD,ntimestep+1,i), ctm.tensorAs().at(i));

            for(auto i:range(2*nsite))
                writeToFile(format("FU_Links_%d_%d_%d", bondD,ntimestep+1,i), ctm.linkAs().at(i));

//                writeToFile(format("As_%d_%d", bondD,ntimestep+1), ctm.tensorAs());
//                writeToFile(format("Links_%d_%d", bondD, ntimestep+1), ctm.linkAs());

            iterout << bondD << "  " << chi << "  " << snorm << "  " << ntimestep + 1
                    << "  " << abs(mztot) << "  " << energytot << "  " << dmz << "  " << denergy
                    << "  " << dFU << "  " << dCTM << endl;
            if ((abs(dmz) < 1.0e-4) && (abs(denergy) < 1.0e-6)) break;
        }
    }

    dCTM = ctm.CTMRG();

    mz.clear();
    mztot = 0.0;
    for (auto i:range(nsite)) {
        auto line = i / nx;
        auto col = i % nx;
        mz.push_back(ctm.exp_site(sites.op("N", i + 1), i));
        cout << "FU results: mz at (line,col) = ( "
             << line << " , " << col << " ) equals " << mz[i] << endl;
        mztot += mz[i];
    }
    mztot /= nsite;

    energy.clear();
    for (auto i:range(2 * nsite)) {
        auto abInd = calcABInd(i, nx, ny, uc);
        auto hh = hab(abInd[0],abInd[1]);

        energy.push_back(ctm.exp_bond(hh, i));
        cout << "FU results: energy at bond " << i << " ) equals " << energy[i] << endl;
    }
    energytot = accumulate(energy.begin(), energy.end(), 0.0) / nsite;

    snorm = ctm.stateNorm();
    mzout << bondD << "  " << chi << "  " << snorm << "  " << ntimestep + 1 << "  "
          << mztot << "  " << energytot << "  " << dFU << "  " << dCTM << endl;

    mzout.close();

    iterout.close();

    return 0;
}