//
// Created by xich on 16-11-25.
//
#include <iostream>
#include <fstream>


#include "iPEPS/iPEPS.h"
#include "iPEPS/CTM.h"

using namespace std;
using namespace itensor;

int main() {
    const int nx = 2;
    const int ny = 2;
    const int nsite=2;
    auto sites = SpinHalf(nsite);
    int cfg[nx*ny]={0, 1, 1, 0};
    vector<int> uc(cfg, cfg + nx * ny);

    int dim = 4;
    int chi = 10;
    Real tstep = 0.01;
    Real lambda = 1.0;
    int maxTimeStep = 4000;

    iPEPS_IQTensor ipeps(uc, sites, nsite, dim, nx, ny);

    auto Eold = 100.0;
    int ntimestep = 0;
    for (ntimestep = 0; ntimestep < maxTimeStep; ++ntimestep) {
        auto Enew = 0.0;
//        cout<<"time step: "<<ntimestep<<endl;
        for (auto i:range(2 * nsite)) {
            auto abInd = calcABInd(i, nx, ny, uc);
//            cout<<"update gate A: "<<abInd.first<<"B:  "<<abInd.second<<endl;
            IQTensor hh = lambda * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
            hh += lambda * 0.5 * sites.op("S+", abInd[0] + 1) * sites.op("S-", abInd[1] + 1);
            hh += lambda * 0.5 * sites.op("S-", abInd[0] + 1) * sites.op("S+", abInd[1] + 1);
//    PrintData(hh);
            auto G = expHermitian(hh, -tstep);
            double e0 = ipeps.applyGate(G, i);
            Enew -= log(e0) / tstep / 2.0;
        }

        if (ntimestep>400 && (Enew - Eold) < 1.0e-8) {
            cout << "simple update converged at time step:" << ntimestep + 1
                 << ", with E=" << Enew / nsite << endl;
            break;
        }

        Eold = Enew;
        if ((ntimestep + 1) % 100 == 0)cout << "n=" << ntimestep + 1 << ", E=" << Enew / nsite << endl;
    }
    if (ntimestep >= maxTimeStep - 1) cout << "Maximun time step reached" << endl;

    ipeps.AfromGandLs();
    CTM_IQTensor ctm(uc, ipeps.tensorAs(), ipeps.linkAs(), nsite,nx,ny,chi);
    ctm.CTMRG();

    vector<Real> mz;
    for(auto i:range(nsite))
    {
        auto line = i/nx;
        auto col = i%nx;
        mz.push_back(ctm.exp_site( sites.op("Sz", i+1), i));
        cout<<"SU results: mz at (line,col) = ( "<<line<<" , "<<col<<" ) equals "<<mz[i]<<endl;
    }

    vector<Real> energy;
    for (auto i:range(2 * nsite)) {
        auto abInd = calcABInd(i, nx, ny, uc);

        IQTensor hh = lambda * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
        hh += lambda * 0.5 * sites.op("S+", abInd[0] + 1) * sites.op("S-", abInd[1] + 1);
        hh += lambda * 0.5 * sites.op("S-", abInd[0] + 1) * sites.op("S+", abInd[1] + 1);

        energy.push_back(ctm.exp_bond( hh, i));
        cout<<"SU results: energy at bond "<<i<<" ) equals "<<energy[i]<<endl;
    }
    cout<<"SU results: total energy = "<<accumulate(energy.begin(),energy.end(),0.0)/nsite<<endl;

// Full update
    Real dmzsave, desave;
    Real timeprop = 0.0;
    Real mzold=mz[0], eold=energy[0];
    Real mznew, enew;
    for (ntimestep = 0; ntimestep < maxTimeStep; ++ntimestep) {
        timeprop += tstep;
        auto Enew = 0.0;
        Real dCTM, dFU;
//        cout<<"time step: "<<ntimestep<<endl;
        for (auto i:range(2 * nsite)) {
            auto abInd = calcABInd(i, nx, ny, uc);
            IQTensor hh = lambda * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
            hh += lambda * 0.5 * sites.op("S+", abInd[0] + 1) * sites.op("S-", abInd[1] + 1);
            hh += lambda * 0.5 * sites.op("S-", abInd[0] + 1) * sites.op("S+", abInd[1] + 1);
//    PrintData(hh);
            auto G = expHermitian(hh, -tstep);

//            dCTM = ctm.CTMRG();
//            dFU = ctm.reduceUpdate(G, i);
            dFU = ctm.fastUpdate(G, i);
            dCTM = ctm.CTMRG();
        }
        auto dmz = abs(mznew) - mzold;
        mzold = abs(mznew);
        if (ntimestep == 10) { dmzsave = dmz; }
        if ((ntimestep) > 40 && abs(dmz / dmzsave) < 0.01) break;
        if ((ntimestep) > 100 && abs(dmz) < 1.0e-8) break;
    }
    ctm.CTMRG();

    mz.clear();
    for(auto i:range(nsite))
    {
        auto line = i/nx;
        auto col = i%nx;
        mz.push_back(ctm.exp_site( sites.op("Sz", i+1), i));
        cout<<"FU results: mz at (line,col) = ( "<<line<<" , "<<col<<" ) equals "<<mz[i]<<endl;
    }

    energy.clear();
    for (auto i:range(2 * nsite)) {
        auto abInd = calcABInd(i, nx, ny, uc);

        IQTensor hh = lambda * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
        hh += lambda * 0.5 * sites.op("S+", abInd[0] + 1) * sites.op("S-", abInd[1] + 1);
        hh += lambda * 0.5 * sites.op("S-", abInd[0] + 1) * sites.op("S+", abInd[1] + 1);

        energy.push_back(ctm.exp_bond( hh, i));
        cout<<"FU results: energy at bond "<<i<<" ) equals "<<energy[i]<<endl;
    }
    cout<<"FU results: total energy = "<<accumulate(energy.begin(),energy.end(),0.0)/nsite<<endl;

    return 0;
}