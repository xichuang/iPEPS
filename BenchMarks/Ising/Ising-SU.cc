//
// Created by xich on 16-10-31.
//
#include <iostream>
#include <fstream>

#include "iPEPS/iPEPS.h"
#include "iPEPS/CTM.h"

using IndexT = Index;
using TensorT = ITensorT<IndexT>;
using iPEPS_TensorT  = iPEPS<IndexT>;
using CTM_TensorT  = CTM<IndexT>;

int main() {
    const int nx = 2;
    const int ny = 2;
    const int nsite=2;
    auto sites = SpinHalf(nsite);
    int cfg[nx*ny]={0,1,1,0};
    vector<int> uc(cfg, cfg + nx * ny);

    int maxTimeStep = 4000;
    Real tstep = 0.005;
    int dim = 3;
    int chi = dim * 10;

    fstream mzout, iterout;
    mzout.open("SU-mz.txt", ios::out);
    iterout.open("SU-iter.txt", ios::out);
    assert(mzout.is_open() && iterout.is_open());

    mzout<< "nlambda" <<"  "<<"lambda"<<"  "<<"norm"<<"  "<< "mz" <<"  "<< "energy" << endl;

    iterout <<"nlambda"<<"  "<<"lambda"<<"  "<<"timestep"<<"  "<< "energy"<< endl;

    for(auto nl=0;nl<50;nl++) {
        Real lambda = 2.7+0.8/50.0*nl;

        cout << "simple Update" << endl;
        iPEPS_TensorT ipeps(uc, sites, nsite, dim, nx, ny);
        auto Eold = 100.0;
        auto Enew = 0.0;
        int ntimestep = 0;
        for (ntimestep = 0; ntimestep < maxTimeStep; ++ntimestep) {

            for(auto i:range(nsite)) {
                TensorT hf = -2.0*lambda * sites.op("Sx", i+1);
                TensorT gF = expHermitian(hf, -tstep / 2.0);
                ipeps.applySiteOp(gF, i);
            }

            Enew=0.0;
            for (auto i:range(2 * nsite))
            {
                auto abInd = calcABInd(i, nx, ny, uc);
                TensorT hi = -4.0 * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
                TensorT gI=expHermitian(hi,-tstep);
                auto e0 = ipeps.applyGate(gI, i);
                Enew -= log(e0);
            }

            for(auto i:range(nsite)) {
                TensorT hf = -2.0*lambda * sites.op("Sx", i+1);
                TensorT gF = expHermitian(hf, -tstep / 2.0);
                ipeps.applySiteOp(gF, i);
            }

            Enew/= 2.0*tstep;
            if ( (abs(Enew - Eold) < 1.0e-8 )&& (ntimestep>400) ) {
                cout << "simple update converged at time step:" << ntimestep + 1
                     << ", with E=" << Enew / nsite << endl;
                break;
            }
            Eold = Enew;
            if ((ntimestep + 1) % 200 == 0) iterout <<nl<<"  "<<lambda
                    <<"  " << ntimestep + 1 << "  " << Enew / 2.0 << endl;
        }
        if (ntimestep >= maxTimeStep - 1) cout << "Maximun SU time step reached" << endl;

        ipeps.AfromGandLs();
        CTM_TensorT ctm(uc, ipeps.tensorAs(), ipeps.linkAs(), ipeps.symInd(),
                        nsite,nx,ny,chi);
        ctm.CTMRG();

        vector<Real> mz;
        for(auto i:range(nsite))
        {
            auto line = i/nx;
            auto col = i%nx;
            mz.push_back(ctm.exp_site( sites.op("Sz", i+1), i));
            cout<<"SU results: mz at (line,col) = ( "<<line<<" , "<<col<<" ) equals "<<mz[i]<<endl;
        }
        auto mztot = accumulate(mz.begin(),mz.end(),0.0)/nsite;
        cout<<"SU results: mz = "<< mztot <<endl;

        vector<Real> energy;
        for (auto i:range(2 * nsite)) {
            auto abInd = calcABInd(i, nx, ny, uc);

            TensorT hh = -4.0 * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
            hh -= lambda * 0.5 * TensorT(sites.op("Sx", abInd[0] + 1)) * sites.op("Id", abInd[1] + 1);
            hh -= lambda * 0.5 * sites.op("Id", abInd[0] + 1) * TensorT(sites.op("Sx", abInd[1] + 1));

            energy.push_back(ctm.exp_bond( hh, i));
            cout<<"SU results: energy at bond "<<i<<" ) equals "<<energy[i]<<endl;
        }
        auto energytot = accumulate(energy.begin(),energy.end(),0.0)/nsite;
        cout<<"SU results: total energy = "<< energytot <<endl;

        Real snorm = ctm.stateNorm();
        mzout<< nl <<"  "<<lambda<<"  "<<snorm<<"  "<< mztot <<"  " << energytot << endl;
    }
    mzout.close();
    iterout.close();
    return 0;
}
