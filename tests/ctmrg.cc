//
// Created by xich on 16-11-25.
//
#include <iostream>
#include <fstream>


#include "iPEPS/iPEPS.h"
#include "iPEPS/CTM.h"
#include "utils/pstrickCTM.h"
#include <sys/time.h>

using namespace std;
using namespace itensor;

int main() {
    struct timeval start1,start2,start3,start4;
    struct timeval end1,end2,end3,end4;

    const int nx = 2;
    const int ny = 2;
    const int nsite=2;
    auto sites = SpinHalf(nsite);
    int cfg[nx*ny]={0,1,1,0};
    vector<int> uc(cfg, cfg + nx * ny);

    int dim = 6;
    int chi = 60;
    Real tstep = 0.01;
    Real lambda = 1.0;
    int maxTimeStep = 2000;

    iPEPS_IQTensor ipeps(uc, sites, nsite, dim, nx, ny);
    gettimeofday(&start1,NULL);
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
        Eold = Enew;
        if ((ntimestep + 1) % 100 == 0)cout << "n=" << ntimestep + 1 << ", E=" << Enew / nsite << endl;
    }
    if (ntimestep >= maxTimeStep - 1) cout << "Maximun time step reached" << endl;
    gettimeofday(&end1,NULL);
    cout<<"SU time:"
        <<1000000 * (end1.tv_sec - start1.tv_sec) + (end1.tv_usec - start1.tv_usec)<<endl;

    ipeps.AfromGandLs();

    CTM_IQTensor ctm(uc, ipeps.tensorAs(), ipeps.linkAs(), nsite,nx,ny,chi);
    gettimeofday(&start1,NULL);
    ctm.CTMRG(10);
    gettimeofday(&end1,NULL);
    cout<<"CTMRG time:"
        <<1000000 * (end1.tv_sec - start1.tv_sec) + (end1.tv_usec - start1.tv_usec)<<endl;
    auto cbs = ctm.combAAs();
    auto aas = ctm.tensorAAs();
    for(auto ni:range(nx*ny)) {
//        cout<<"si: "<< ni<<endl;
        auto env = ctm.returnEnv(ni);
        auto bulk = ctm.returnBulk(ni);

        int line = ni/nx;
        int col = ni%nx;
        vector<IQIndex> bulkLink;
        auto ind=indLC(line,col,nx,ny);
        auto sb = siteBonds(uc[ind],nx,ny,uc);
        bulkLink.push_back(dag(commonIndex( aas[uc[ind]], cbs[sb[2]])));
        bulkLink.push_back(dag(commonIndex( aas[uc[ind]], cbs[sb[3]])));
        ind=indLC(line+1,col+1,nx,ny);
        sb = siteBonds(uc[ind],nx,ny,uc);
        bulkLink.push_back(commonIndex( aas[uc[ind]], cbs[sb[0]]));
        bulkLink.push_back(commonIndex( aas[uc[ind]], cbs[sb[1]]));
/*
        for (auto &envT:env)
            Print(envT);
        for (auto &bulkT:bulk)
            Print(bulkT);
*/
        pstrickCTM<IQIndex>(env, bulk, bulkLink, nameint("ctm_",ni));
    }

    return 0;
}

