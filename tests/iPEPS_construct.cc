//
// Created by xich on 16-11-25.
//
#include <iostream>
#include <fstream>

#include "iPEPS/iPEPS.h"

using namespace std;
using namespace itensor;

int main() {
    int dim = 3;
    int chi = 10;
    const int nx = 2;
    const int ny = 2;
    const int nsite=4;

    auto sites = SpinHalf(nsite);

    int cfg[nx*ny]={0,1,2,3};
    vector<int> uc(cfg, cfg + nx * ny);

    for(auto i:range(2*nsite)) {
        auto indAB = calcABInd(i, nx, ny, uc);
        cout << i << " A: " << indAB[0]<<" B: "<<indAB[1]<<endl;
    }

    iPEPS_ITensor ipeps(uc, sites, nsite, dim, nx, ny);
    ipeps.AfromGandLs();
    auto iA = ipeps.tensorAs();
    auto iLink = ipeps.linkAs();
    for(auto &I:iLink)
        Print(I);
    for(auto &tA:iA)
        Print(tA);

    cout<<"IQTensor:"<<endl;
    iPEPS_IQTensor iIQpeps(uc, sites, nsite, dim, nx, ny);
    iIQpeps.AfromGandLs();
    auto iqA = iIQpeps.tensorAs();
    auto iqLink = iIQpeps.linkAs();
    for(auto &I:iqLink)
        Print(I);
    for(auto &tA:iqA)
        Print(tA);

    return 0;
}

