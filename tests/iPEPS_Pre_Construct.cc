//
// Created by xich on 16-11-25.
//
#include <iostream>
#include <fstream>

#include "iPEPS/iPEPS.h"
#include "iPEPS/CTM.h"

using namespace std;
using namespace itensor;

void initialGs(const int nSite, const int nx, const int ny,
               const vector<int> unitCell,
               const SiteSet& sites,
               vector< array<IQIndex,2> > &linksG,
               vector<IQTensor> &Ls, vector<IQTensor> &G){
    linksG.clear();
    for (auto i:range(nSite)) {
        linksG.push_back(
                array<IQIndex, 2>{{IQIndex(nameint("UpIn", i),
                                           Index("q-1", 1, Link), QN(-1),
                                           Index("q0", 2, Link), QN(0),
                                           Index("q1", 1, Link), QN(1), In),
                                          IQIndex(nameint("UpOut", i),
                                                  Index("q-1", 1, Link), QN(-1),
                                                  Index("q0", 2, Link), QN(0),
                                                  Index("q1", 1, Link), QN(1) )}});
        linksG.push_back(
                array<IQIndex, 2>{{IQIndex(nameint("LeftIn", i),
                                           Index("q-1", 1, Link), QN(-1),
                                           Index("q0", 2, Link), QN(0),
                                           Index("q1", 1, Link), QN(1), In),
                                          IQIndex(nameint("LeftOut", i),
                                                  Index("q-1", 1, Link), QN(-1),
                                                  Index("q0", 2, Link), QN(0),
                                                  Index("q1", 1, Link), QN(1) )}});
    }
    Ls.clear();
    for (auto &I:linksG) {
        Ls.push_back(delta(I[0], I[1]));
    }
    G.clear();
    for (auto i:range(nSite)) {
        auto sitebond = siteBonds(i, nx, ny, unitCell);
        G.push_back(randomTensor(QN(),
                                  dag(linksG[sitebond[0]][0]),
                                  dag(linksG[sitebond[1]][1]),
                                  dag(linksG[sitebond[2]][1]),
                                  dag(linksG[sitebond[3]][0]),
                                  sites.si(i + 1)));
        scaleTensor(G[i]);
    }
}

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

    vector< array<IQIndex,2> > linksG;
    vector<IQTensor> Ls,G;
    initialGs(nsite,nx,ny,uc,sites,linksG,Ls,G);

    iPEPS_IQTensor ipeps(uc,sites,linksG,Ls,G,nsite,dim,nx,ny);

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
        if (ntimestep>1000 && (Enew - Eold) < 1.0e-8) {
            cout << "simple update converged at time step:" << ntimestep + 1
                 << ", with E=" << Enew / nsite << endl;
            break;
        }
        if (ntimestep>1000 && (Enew - Eold) < 1.0e-8) {
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

    return 0;
}

