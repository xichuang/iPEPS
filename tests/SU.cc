//
// Created by xich on 16-11-25.
//
#include <iostream>
#include <fstream>


#include "iPEPS/iPEPS.h"

using namespace std;
using namespace itensor;

int main() {
    const int nx = 2;
    const int ny = 2;
    const int nsite=4;
    auto sites = SpinHalf(nsite);
    int cfg[nx*ny]={0,1,2,3};
    vector<int> uc(cfg, cfg + nx * ny);

    int dim = 4;
    int chi = 10;
    Real tstep = 0.01;
    Real lambda = 1.0;
    int maxTimeStep = 4000;

    iPEPS_ITensor ipeps(uc, sites, nsite, dim, nx, ny);

    auto Eold = 100.0;
    int ntimestep = 0;
    for (ntimestep = 0; ntimestep < maxTimeStep; ++ntimestep) {
        auto Enew = 0.0;
//        cout<<"time step: "<<ntimestep<<endl;
        for (auto i:range(2 * nsite)) {
            auto abInd = calcABInd(i, nx, ny, uc);
//            cout<<"update gate A: "<<abInd.first<<"B:  "<<abInd.second<<endl;
            ITensor hh = lambda * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
            hh += lambda * 0.5 * sites.op("S+", abInd[0] + 1) * sites.op("S-", abInd[1] + 1);
            hh += lambda * 0.5 * sites.op("S-", abInd[0] + 1) * sites.op("S+", abInd[1] + 1);
//    PrintData(hh);
            auto G = expHermitian(hh, -tstep);
            double e0 = ipeps.applyGate(G, i);
            Enew -= log(e0) / tstep / 2.0;
        }
        Eold = Enew;
        if ((ntimestep + 1) % 100 == 0)cout << "n=" << ntimestep + 1 << ", E=" << Enew / nsite << endl;
    }
    if (ntimestep >= maxTimeStep - 1) cout << "Maximun time step reached" << endl;

    return 0;
}

