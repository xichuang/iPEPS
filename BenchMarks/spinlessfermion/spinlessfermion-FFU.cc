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

void initialGs(const int nSite, const int nx, const int ny,
               const vector<int> unitCell, const SiteSet &sites,
               vector<array<IQIndex, 2> > &linksG, vector<IQTensor> &Ls,
               vector<IQTensor> &G, IQIndex &symInd) {
    linksG.clear();

    auto qn0 = QN("Pf",1);
    auto symTen = dag(symmtensor(qn0));
    symInd = symTen.inds().front();

    for (auto i:range(nSite)) {
        linksG.push_back(
                array<IQIndex, 2>{{IQIndex(nameint("UpIn", i),
                                           Index("q0", 2, Link), QN("Pf", 0),
                                           Index("q1", 2, Link), QN("Pf", 1), In),
                                          IQIndex(nameint("UpOut", i),
                                                  Index("q0", 2, Link), QN("Pf", 0),
                                                  Index("q1", 2, Link), QN("Pf", 1))}});
        linksG.push_back(
                array<IQIndex, 2>{{IQIndex(nameint("LeftIn", i),
                                           Index("q0", 2, Link), QN("Pf", 0),
                                           Index("q1", 2, Link), QN("Pf", 1), In),
                                          IQIndex(nameint("LeftOut", i),
                                                  Index("q0", 2, Link), QN("Pf", 0),
                                                  Index("q1", 2, Link), QN("Pf", 1))}});
    }
    Ls.clear();
    for (auto &I:linksG) {
        Ls.push_back(delta(I[0], I[1]));
    }
    G.clear();
    for (auto i:range(nSite)) {
        auto sitebond = siteBonds(i, nx, ny, unitCell);
        auto qn = QN();
//        if(i%2==0) qn=QN("Nf",1);
        if (i == 0) qn = qn0;
        auto Gtmp = randomTensor(qn,
                                 dag(linksG[sitebond[0]][0]),
                                 dag(linksG[sitebond[1]][1]),
                                 dag(linksG[sitebond[2]][1]),
                                 dag(linksG[sitebond[3]][0]),
                                 sites.si(i + 1));
        if(i==0) Gtmp *= symTen;
        scaleTensor(Gtmp);
        G.push_back(Gtmp);
    }
}

int main() {
    fstream mzout, iterout;
    mzout.open("meanN.txt", ios::out);
    iterout.open("iter.txt", ios::out);
    assert(mzout.is_open() && iterout.is_open());
    int maxTimeStep = 8000;
    Real tstep = 0.005;
    int dim = 4;
    int chi = dim * 10;

    const int nx = 2;
    const int ny = 2;
    const int nsite = 2;
    auto sites = Spinless(nsite,{"ConserveNf",false});
    writeToFile(format("sites_%d", nsite), sites);
    int cfg[nx * ny] = {0, 1, 1, 0};
    vector<int> uc(cfg, cfg + nx * ny);
    vector< array<IQIndex,2> > linksG;
    vector<IQTensor> Ls,G;
    IQIndex symInd;
    initialGs(nsite, nx, ny, uc, sites, linksG, Ls, G, symInd);
    iPEPS_IQTensor ipeps(uc, sites, linksG, Ls, G, symInd,
                         nsite, dim, nx, ny, {"Fermionic", true});

    Real lambda = 1.0,VV=0.0,mu=0.4,dD=0.2;
    auto hab=[&](const int i, const int j)
    {
        TensorT hh=sites.op("Adag", i + 1) * sites.op("A", j + 1);
        hh += sites.op("A", i + 1) * sites.op("Adag", j + 1);
        hh -= dD * sites.op("A", i + 1) * sites.op("A", j + 1);
        hh -= dD * sites.op("Adag", i + 1) * sites.op("Adag", j + 1);
        hh -= 0.25 *mu *sites.op("N", i + 1) * sites.op("Id", j + 1);
        hh -= 0.25 *mu *sites.op("Id", i + 1) * sites.op("N", j + 1);
        hh += VV *sites.op("N", i + 1) * sites.op("N",  j+ 1);

        return hh;
    };

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
            cout << "simple update at time step:" << ntimestep + 1
                 << ", with E=" << Enew / nsite << endl;
            if ( abs(Enew - Eold) < 1.0e-8 ) {
                cout << "simple update converged at time step:" << ntimestep + 1
                     << ", with E=" << Enew / nsite << endl;
                break;
            }
            Eold = Enew;
        }
    }
    if (ntimestep >= maxTimeStep - 1) cout << "Maximun SU time step reached, with E="
                                           << Enew / 2.0 << endl;

    ipeps.AfromGandLs();
    CTM_TensorT ctm(uc, ipeps.tensorAs(), ipeps.linkAs(), ipeps.symInd(),
                    nsite, nx, ny, chi,{"Fermionic",true});

    for(auto i:range(nsite))
        writeToFile(format("SU_As_%d_%d", ntimestep+1,i), ctm.tensorAs().at(i));

    for(auto i:range(2*nsite))
        writeToFile(format("SU_Links_%d_%d", ntimestep+1,i), ctm.linkAs().at(i));

    ctm.CTMRG();

    vector<Real> mz;
    for (auto i:range(nsite)) {
        auto line = i / nx;
        auto col = i % nx;
        mz.push_back(ctm.exp_site(sites.op("N", i + 1), i));
        cout << "SU results: mz at (line,col) = ( " << line << " , " << col << " ) equals " << mz[i] << endl;
    }
    auto mztot = accumulate(mz.begin(),mz.end(),0.0)/nsite;
    cout << "SU results: meanN = " << mztot << endl;

    vector<Real> energy;
    for (auto i:range(2 * nsite)) {
        auto abInd = calcABInd(i, nx, ny, uc);
        auto hh = hab(abInd[0],abInd[1]);

        energy.push_back(ctm.exp_bond(hh, i));
        cout << "SU results: energy at bond " << i << " ) equals " << energy[i] << endl;
    }
    auto energytot = accumulate(energy.begin(), energy.end(), 0.0) / nsite;
    cout << "SU results: total energy = " << energytot << endl;
    Real snorm = ctm.stateNorm()  ;


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
            cout<<"dim = "<<bondD<<", time step = "<<ntimestep<<endl;
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
//            dCTM += dCTM0;
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
//            dCTM += dCTM0;
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

//            dCTM += dCTM0;
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
                    writeToFile(format("As_%d_%d_%d", bondD,ntimestep+1,i), ctm.tensorAs().at(i));

                for(auto i:range(2*nsite))
                    writeToFile(format("Links_%d_%d_%d", bondD,ntimestep+1,i), ctm.linkAs().at(i));

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
    }
    mzout.close();
    iterout.close();

    return 0;
}