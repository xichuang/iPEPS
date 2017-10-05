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
    auto dD = input.getReal("dD");

    auto maxTimeStep=input.getInt("maxTimeStep");
    auto tstep = input.getReal("tstep");
    auto mindim = input.getInt("mindim");
    auto maxdim = input.getInt("maxdim");
    auto repeat = input.getInt("repeat");

    fstream mzout, iterout;
    mzout.open("meanN.txt", ios::out);
    iterout.open("iter.txt", ios::out);
    assert(mzout.is_open() && iterout.is_open());
    mzout <<"nrepeat"<<"  "<< "bondD" << "  " << "chi" << "  " << "mztot" << "  "
          << "energytot" << "  " << "snorm" << "  " << "dctm"<< endl;

    // set unitcell
    const int nx = 2;
    const int ny = 2;
    const int nsite = 2;
    auto sites = Spinless(nsite, {"ConserveNf", false});
    writeToFile("SU_site",sites);
    int cfg[nx * ny] = {0, 1, 1, 0};
    vector<int> uc(cfg, cfg + nx * ny);

    //build iPEPS with inital tensors.
    vector<array<IQIndex, 2> > linksG;
    vector<IQTensor> Ls, G;
    IQIndex symInd;
    initialGs(nsite, nx, ny, uc, sites, linksG, Ls, G, symInd);
    iPEPS_IQTensor ipeps(uc, sites, linksG, Ls, G, symInd,
                         nsite, mindim, nx, ny, {"Fermionic", true});

    //define Hamitonian;
    auto hab = [&](const int i, const int j) {
        TensorT hh = sites.op("Adag", i + 1) * sites.op("A", j + 1);
        hh += sites.op("A", i + 1) * sites.op("Adag", j + 1);
        hh -= dD * sites.op("A", i + 1) * sites.op("A", j + 1);
        hh -= dD * sites.op("Adag", i + 1) * sites.op("Adag", j + 1);
        hh -= 0.25 * mu * sites.op("N", i + 1) * sites.op("Id", j + 1);
        hh -= 0.25 * mu * sites.op("Id", i + 1) * sites.op("N", j + 1);
        hh += VV * sites.op("N", i + 1) * sites.op("N", j + 1);

        return hh;
    };

    auto Eold = 100.0;
    auto Enew = 0.0;

    int nrepeat = 0;
    Real dctmOld=10.0;
    while(nrepeat<repeat) {
        cout<<"repeat number = "<< nrepeat<<endl;
        ipeps.randomGs();
        for (int bondD = mindim; bondD <= maxdim; ++bondD) {
            ipeps.resetBondD(bondD);
            auto chi = bondD * 10;
            int ntimestep = 0;
            for (ntimestep = 0; ntimestep < maxTimeStep; ++ntimestep) {
                Enew = 0.0;
                for (auto i:range(2 * nsite - 1)) {
                    auto abInd = calcABInd(i, nx, ny, uc);
                    auto hh = hab(abInd[0], abInd[1]);
                    auto G = expHermitian(hh, -tstep / 2.0);
                    auto e0 = ipeps.applyGate(G, i);
                    Enew -= log(e0) / tstep;
                }
                {
                    auto i = 2 * nsite - 1;
                    auto abInd = calcABInd(i, nx, ny, uc);
                    auto hh = hab(abInd[0], abInd[1]);
                    auto G = expHermitian(hh, -tstep);
                    auto e0 = ipeps.applyGate(G, i);
                    Enew -= log(e0) / (2.0 * tstep);
                }
                for (auto j:range(2 * nsite - 1)) {
                    auto i = 2 * nsite - j - 2;
                    auto abInd = calcABInd(i, nx, ny, uc);
                    auto hh = hab(abInd[0], abInd[1]);
                    auto G = expHermitian(hh, -tstep / 2.0);
                    auto e0 = ipeps.applyGate(G, i);
                    Enew -= log(e0) / tstep;
                }

                if ((ntimestep + 1) % 100 == 0) {
                    cout << "simple update at time step:" << ntimestep + 1
                         << ", with E=" << Enew / nsite << endl;
                    iterout << bondD << "  " << ntimestep + 1 << "  " << Enew / nsite << endl;
                    if (abs(Enew - Eold) < 1.0e-8) {
                        cout << "simple update converged at time step:" << ntimestep + 1
                             << ", with E=" << Enew / nsite << endl;
                        break;
                    }
                    Eold = Enew;
                }
            }
            if (ntimestep >= maxTimeStep - 1)
                cout << "Maximun SU time step reached, with E="
                     << Enew / 2.0 << endl;

            ipeps.AfromGandLs();
            CTM_TensorT ctm(uc, ipeps.tensorAs(), ipeps.linkAs(), ipeps.symInd(),
                            nsite, nx, ny, chi, {"Fermionic", true});
//            ctm.randEnv();
            auto dctm = ctm.CTMRG();
            auto As=ctm.tensorAs();
            auto Links=ctm.linkAs();
            for(auto i:range(nsite))
                writeToFile(format("SU_As_%d_%d_%d",nrepeat,bondD,i),As[i]);

            for(auto i:range(2*nsite))
                writeToFile(format("SU_Links_%d_%d_%d",nrepeat,bondD,i),Links[i]);
/*
            if(dctm<dctmOld){
                for(auto i:range(nsite))
                    writeToFile(format("SU_As_%d_%d", bondD,i),As[i]);
                    writeToFile(format("SU_Links_%d_%d_%d",nrepeat,bondD,i),Links[i]);
            }
            dctmOld=dctm;
            */

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

            mzout <<nrepeat<<"  "<< bondD << "  " << chi << "  " << mztot
                  << "  " << energytot << "  " << snorm << "  " << dctm << endl;
        }

        ++nrepeat;
    }

    mzout.close();
    iterout.close();

    return 0;
}