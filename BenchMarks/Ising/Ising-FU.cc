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
    mzout.open("FU-mz.txt", ios::out);
    iterout.open("FU-iter.txt", ios::out);
    assert(mzout.is_open() && iterout.is_open());

    mzout<< "nlambda" <<"  "<<"lambda"<<"  "<<"norm"<<"  "<< "ntimestep"<<"  "<<"mz" <<"  "<< "energy"
         <<"  "<<"dFU"<<"  "<<"dCTM" << endl;

    iterout <<"nlambda"<<"  "<<"lambda"<<"  "<<"norm"<<"  "<<"timestep"
            <<"  "<< "mz"<<"  "<< "energy"<< "  "<< "dmz"<<"  "<< "denergy"
            <<"  "<<"dFU"<<"  "<<"dCTM"<<endl;

    auto minlambda = 2.7;
    auto addlambda = 0.8; //max lambda = minlambda+addlambda
    auto nlambda = 50;

    //##############################################
    // SU as initial state.
    Real lambda = minlambda;

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
                 << ", with E=" << Enew / 2.0 << endl;
            break;
        }
        Eold = Enew;
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
    }
    auto mztot = accumulate(mz.begin(),mz.end(),0.0)/nsite;

    vector<Real> energy;
    for (auto i:range(2 * nsite)) {
        auto abInd = calcABInd(i, nx, ny, uc);

        TensorT hh = -4.0 * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
        hh -= lambda * 0.5 * TensorT(sites.op("Sx", abInd[0] + 1)) * sites.op("Id", abInd[1] + 1);
        hh -= lambda * 0.5 * sites.op("Id", abInd[0] + 1) * TensorT(sites.op("Sx", abInd[1] + 1));

        energy.push_back(ctm.exp_bond( hh, i));
    }
    auto energytot = accumulate(energy.begin(),energy.end(),0.0)/nsite;
    Real snorm = ctm.stateNorm();

    //##############################################
    //FFU
    for(auto nl=0;nl<nlambda;nl++) {
        Real lambda = minlambda+addlambda/nlambda*nl;
        Real mzold = abs(mztot);
        Real energyold = energytot;
        Real timeprop = 0.0;

        Real dCTM=0.0;
        Real dFU=0.0;
        for (ntimestep = 0; ntimestep < maxTimeStep; ++ntimestep) {
            timeprop += tstep;

            for(auto i:range(nsite)) {
                TensorT hf = -2.0*lambda * sites.op("Sx", i+1);
                TensorT gF = expHermitian(hf, -tstep / 2.0);
                ctm.applySiteOp(gF, i);
            }

            dCTM=0.0;
            dFU=0.0;
            for (auto i:range(2 * nsite))
            {
                auto abInd = calcABInd(i, nx, ny, uc);
                TensorT hi = -4.0 * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
                TensorT gI=expHermitian(hi,-tstep);

                auto dCTM0 = ctm.CTMRG();
                auto dFU0 = ctm.reduceUpdate(gI, i);
//                auto dFU0 = ctm.fastUpdate(gI,i);

                dCTM += dCTM0;
                dFU += dFU0;
            }

            for(auto i:range(nsite)) {
                TensorT hf = -2.0*lambda * sites.op("Sx", i+1);
                TensorT gF = expHermitian(hf, -tstep / 2.0);
                ctm.applySiteOp(gF, i);
            }

//       writeToFile(format("./IsingDat/Aat_%d", ntimestep), ctm.tensorA());
//        writeToFile(format("./IsingDat/Bat_%d", ntimestep), ctm.tensorB());
//        writeToFile(format("./IsingDat/linksat_%d", ntimestep), ctm.linksAB());

            if((ntimestep + 1) % 100 == 0)
            {
                cout << "time step: " << ntimestep+1 << endl;
                ctm.CTMRG();

                mz.clear();
                for(auto i:range(nsite))
                {
                    auto line = i/nx;
                    auto col = i%nx;
                    mz.push_back(ctm.exp_site( sites.op("Sz", i+1), i));
                }
                mztot = accumulate(mz.begin(),mz.end(),0.0)/nsite;

                energy.clear();
                for (auto i:range(2 * nsite)) {
                    auto abInd = calcABInd(i, nx, ny, uc);

                    TensorT hh = -4.0 * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
                    hh -= lambda * 0.5 * TensorT(sites.op("Sx", abInd[0] + 1)) * sites.op("Id", abInd[1] + 1);
                    hh -= lambda * 0.5 * sites.op("Id", abInd[0] + 1) * TensorT(sites.op("Sx", abInd[1] + 1));

                    energy.push_back(ctm.exp_bond( hh, i));
                }
                energytot = accumulate(energy.begin(),energy.end(),0.0)/nsite;

                snorm = ctm.stateNorm();

                auto dmz = abs(mztot) - mzold;
                auto denergy = energytot - energyold;
                mzold=abs(mztot);
                energyold = energytot;

                iterout << nl<<"  "<<lambda<<"  "<<snorm<<"  "<<ntimestep+1
                        <<"  "<<abs(mztot)<<"  "<<energytot<<"  "<<dmz<<"  "<<denergy
                        <<"  "<<dFU<<"  "<<dCTM<<endl;
                if( (abs(dmz) < 1.0e-5) && (abs(denergy)<1.0e-6) ) break;
            }
        }

        ctm.CTMRG();

        mz.clear();
        for(auto i:range(nsite))
        {
            auto line = i/nx;
            auto col = i%nx;
            mz.push_back(ctm.exp_site( sites.op("Sz", i+1), i));
        }
        mztot = accumulate(mz.begin(),mz.end(),0.0)/nsite;

        energy.clear();
        for (auto i:range(2 * nsite)) {
            auto abInd = calcABInd(i, nx, ny, uc);

            TensorT hh = -4.0 * sites.op("Sz", abInd[0] + 1) * sites.op("Sz", abInd[1] + 1);
            hh -= lambda * 0.5 * TensorT(sites.op("Sx", abInd[0] + 1)) * sites.op("Id", abInd[1] + 1);
            hh -= lambda * 0.5 * sites.op("Id", abInd[0] + 1) * TensorT(sites.op("Sx", abInd[1] + 1));

            energy.push_back(ctm.exp_bond( hh, i));
        }
        energytot = accumulate(energy.begin(),energy.end(),0.0)/nsite;

        snorm = ctm.stateNorm();
        mzout<< nl <<"  "<<lambda<<"  "<<snorm<<"  "<<ntimestep+1<<"  "
             << mztot <<"  " << energytot <<"  " <<dFU<<"  "<<dCTM<<endl;

    }
    mzout.close();
    iterout.close();
    return 0;
}
