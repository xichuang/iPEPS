//
// Created by xich on 16-10-31.
//
#include <iostream>
#include <fstream>

#include "iPEPS/iPEPS.h"
#include "iPEPS/CTM.h"

using IndexT = IQIndex;
using TensorT = ITensorT<IndexT>;
using iPEPS_TensorT  = iPEPS<IndexT>;
using CTM_TensorT  = CTM<IndexT>;

int main() {
    const int nx = 2;
    const int ny = 2;
    const int nsite = 2;
    auto sites = SpinHalf(nsite);
    int cfg[nx * ny] = {0, 1, 1, 0};
    vector<int> uc(cfg, cfg + nx * ny);

    int maxTimeStep = 4000;
    Real tstep = 0.005;
    int dim = 4;
    int chi = dim * 10;

    fstream mzout, iterout;
    mzout.open("FFU-mz.txt", ios::out);
    iterout.open("FFU-iter.txt", ios::out);
    assert(mzout.is_open() && iterout.is_open());

    mzout << "dim" << "  " << "chi" << "  " << "norm" << "  " << "ntimestep" << "  " << "mz" << "  " << "energy"
          << "  " << "dFU" << "  " << "dCTM" << endl;

    iterout << "dim" << "  " << "chi" << "  " << "norm" << "  " << "timestep"
            << "  " << "mz" << "  " << "energy" << "  " << "dmz" << "  " << "denergy"
            << "  " << "dFU" << "  " << "dCTM" << endl;

    //##############################################
    // SU as initial state.
    Real lambda = 1.0;

    auto hab=[&](const int i, const int j)
    {
        TensorT hh = lambda * sites.op("Sz", i + 1) * sites.op("Sz", j + 1);
        hh += lambda * 0.5 * sites.op("S+", i + 1) * sites.op("S-", j + 1);
        hh += lambda * 0.5 * sites.op("S-", i + 1) * sites.op("S+", j + 1);

        return hh;
    };

    iPEPS_TensorT ipeps(uc, sites, nsite, dim, nx, ny);
    auto Eold = 100.0;
    auto Enew = 0.0;
    int ntimestep = 0;
    for (ntimestep = 0; ntimestep < maxTimeStep; ++ntimestep) {
        Enew = 0.0;
        for (auto i:range(2 * nsite - 1)) {
            auto abInd = calcABInd(i, nx, ny, uc);
            auto hh = hab(abInd[0],abInd[1]);
            auto G = expHermitian(hh, -tstep / 2.0);
            auto e0 = ipeps.applyGate(G, i);
            Enew -= log(e0) / tstep;
        }
        {
            auto i = 2 * nsite - 1;
            auto abInd = calcABInd(i, nx, ny, uc);
            auto hh = hab(abInd[0],abInd[1]);
            auto G = expHermitian(hh, -tstep);
            auto e0 = ipeps.applyGate(G, i);
            Enew -= log(e0) / (2.0 * tstep);
        }
        for (auto j:range(2 * nsite - 1)) {
            auto i = 2 * nsite - j - 2;
            auto abInd = calcABInd(i, nx, ny, uc);
            auto hh = hab(abInd[0],abInd[1]);
            auto G = expHermitian(hh, -tstep / 2.0);
            auto e0 = ipeps.applyGate(G, i);
            Enew -= log(e0) / tstep;
        }

        if ((ntimestep + 1) % 100 == 0)
        {
            if ( abs(Enew - Eold) < 1.0e-8 ) {
                cout << "simple update converged at time step:" << ntimestep + 1
                     << ", with E=" << Enew / 2.0 << endl;
                break;
            }
            Eold = Enew;
        }
    }
    if (ntimestep >= maxTimeStep - 1) cout << "Maximun SU time step reached, with E="
                                           << Enew / 2.0 << endl;

    ipeps.AfromGandLs();
    CTM_TensorT ctm(uc, ipeps.tensorAs(), ipeps.linkAs(), ipeps.symInd(),
                    nsite, nx, ny, chi);
    ctm.CTMRG();

    vector<Real> mz;
    Real mztot = 0.0;
    for (auto i:range(nsite)) {
        auto line = i / nx;
        auto col = i % nx;
        mz.push_back(ctm.exp_site(sites.op("Sz", i + 1), i));
        mztot += ((i % 2 == 0) ? 1 : -1) * mz[i];
        cout << "SU results: mz at (line,col) = ( " << line << " , " << col << " ) equals " << mz[i] << endl;
    }
    mztot /= nsite;
//        auto mztot = accumulate(mz.begin(),mz.end(),0.0)/nsite;
    cout << "SU results: mz = " << mztot << endl;

    vector<Real> energy;
    for (auto i:range(2 * nsite)) {
        auto abInd = calcABInd(i, nx, ny, uc);
        auto hh = hab(abInd[0],abInd[1]);

        energy.push_back(ctm.exp_bond(hh, i));
        cout << "SU results: energy at bond " << i << " ) equals " << energy[i] << endl;
    }
    auto energytot = accumulate(energy.begin(), energy.end(), 0.0) / nsite;
    cout << "SU results: total energy = " << energytot << endl;
    Real snorm = ctm.stateNorm();


    //##############################################
    //FFU
    for(int bondD=dim;bondD<8;++bondD) {
        ctm.resetBondD(bondD);
        chi = bondD * 10;
        ctm.resetEnvD(chi);
        Real mzold = abs(mztot);
        Real energyold = energytot;
        Real timeprop = 0.0;

        Real dCTM = 0.0;
        Real dFU = 0.0;
        for (ntimestep = 0; ntimestep < maxTimeStep; ++ntimestep) {
            timeprop += tstep;
            dCTM = 0.0;
            dFU = 0.0;
            for (auto i:range(2 * nsite - 1)) {
                auto abInd = calcABInd(i, nx, ny, uc);
                auto hh = hab(abInd[0],abInd[1]);
                auto G = expHermitian(hh, -tstep / 2.0);
//            auto dCTM0 = ctm.CTMRG();
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
//            auto dCTM0 = ctm.CTMRG();
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
//            auto dCTM0 = ctm.CTMRG();
//            auto dFU0 = ctm.reduceUpdate(G, i);
                auto dFU0 = ctm.fastUpdate(G, i);
                auto dCTM0 = ctm.CTMRG();

                dCTM += dCTM0;
                dFU += dFU0;
            }

//       writeToFile(format("./IsingDat/Aat_%d", ntimestep), ctm.tensorA());
//        writeToFile(format("./IsingDat/Bat_%d", ntimestep), ctm.tensorB());
//        writeToFile(format("./IsingDat/linksat_%d", ntimestep), ctm.linksAB());

            if ((ntimestep + 1) % 100 == 0) {
                cout << "time step: " << ntimestep + 1 << endl;
                dCTM = ctm.CTMRG();

                mz.clear();
                mztot = 0.0;
                for (auto i:range(nsite)) {
                    auto line = i / nx;
                    auto col = i % nx;
                    mz.push_back(ctm.exp_site(sites.op("Sz", i + 1), i));
                    mztot += ((i % 2 == 0) ? 1 : -1) * mz[i];
                }
                mztot /= nsite;

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

                iterout << bondD << "  " << chi << "  " << snorm << "  " << ntimestep + 1
                        << "  " << abs(mztot) << "  " << energytot << "  " << dmz << "  " << denergy
                        << "  " << dFU << "  " << dCTM << endl;
                if ((abs(dmz) < 1.0e-5) && (abs(denergy) < 1.0e-6)) break;
            }
        }

        dCTM = ctm.CTMRG();

        mz.clear();
        mztot = 0.0;
        for (auto i:range(nsite)) {
            auto line = i / nx;
            auto col = i % nx;
            mz.push_back(ctm.exp_site(sites.op("Sz", i + 1), i));
            cout << "FU results: mz at (line,col) = ( "
                 << line << " , " << col << " ) equals " << mz[i] << endl;
            mztot += ((i % 2 == 0) ? 1 : -1) * mz[i];
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
    }
    mzout.close();
    iterout.close();
    return 0;
}
