//
// Created by xich on 16-11-17.
//

#include "CTM.h"
#include <algorithm>
#include <ctime>

template<class IndexT>
CTM<IndexT>::CTM(const vector<int> &uc, const vector<TensorT> &As,
                 const vector<IndexT> &links, const IndexT &symInd,
                 const int nsite, const int nx, const int ny, const int chi,
                 const Args &args)
        : unitCell_(uc), nSite_(nsite),
          nx_(nx), ny_(ny), chi_(chi),
          As_(As), links_(links), symInd_(symInd),
          updatedEnv_(-1), constructed_(-1) {
    isFerm_ = args.getBool("Fermionic", false);
    bondD_ = links[0].m();
    combAB_.clear();
    for (auto &ind:links)
        combAB_.push_back(combiner(ind, prime(dag(ind))));

    scaleAs();
    AAfromAs();
    EnvFromAs();
    checkNorm();
}

template<class IndexT>
void CTM<IndexT>::AAfromA(TensorT &AA, const TensorT &A, const int site) {
    auto barA = dag(A);
    auto siteInd = findtype(barA, Site);
    if (isFerm_) {
        auto bonds = siteBonds(site, nx_, ny_, unitCell_);
        fSwapGate(barA, links_[bonds[0]], links_[bonds[1]]);
        fSwapGate(barA, links_[bonds[2]], links_[bonds[3]]);
        barA.primeExcept(symInd_, siteInd);
        AA = A * barA;

        fSwapGate(AA, prime(links_[bonds[0]]), links_[bonds[1]]);
        fSwapGate(AA, links_[bonds[2]], prime(links_[bonds[3]]));
    } else {
        barA.primeExcept(dag(symInd_), dag(siteInd));
        AA = A * barA;
    }
    combineBond(AA, site);
}

template<class IndexT>
void CTM<IndexT>::checkNorm() {

    /*
    for (auto bond:range(2 * nSite_)) {
        bondEnv(bond);
        cout << "Norm for bond " << bond << " equals " << norm_.real() << endl;
        if (norm_.real() < 0.0) {
            auto vert = (bond % 2 == 0);
            abInd_ = calcABInd(bond, nx_, ny_, unitCell_);
            auto si = vert ? abInd_[1] : abInd_[0];
            if(si!=constructed_) constructBulkEnv(si);
            Cs_[envT_[0][0]][envT_[0][1]] = -Cs_[envT_[0][0]][envT_[0][1]];
        }
    }
     */
    for (auto bond:range(2 * nSite_)) {
        bondEnv(bond);
        cout << "New Norm for bond " << bond << " equals " << norm_.real() << endl;
    }

}

template<class IndexT>
void CTM<IndexT>::AAfromAs() {
    AAs_.clear();
    for (auto n:range(nSite_)) {
        TensorT AA;
        AAfromA(AA, As_[n], n);
        AAs_.push_back(AA);
    }
}

// reset envlinks' id to avoid being duplicated with the bulk tensors.
// 0<= bond < 2*nSite_
template<class IndexT>
void CTM<IndexT>::resetEnvLink(const int bond) {
    auto vert = (bond % 2 == 0);
    auto abInd = calcABInd(bond, nx_, ny_, unitCell_);
    if (vert) {
        auto comInd = commonIndex(Cs_[abInd[0]][3], Cs_[abInd[1]][0]);
        auto newbond = cpInd(comInd, {"IndexName", "env", "IndexType", Wtype});
        Cs_[abInd[0]][2] *= delta(dag(comInd), newbond);
        Cs_[abInd[0]][3] *= delta(dag(comInd), newbond);
        Ts_[abInd[0]][1] *= delta(dag(comInd), newbond);
        Ts_[abInd[0]][3] *= delta(dag(comInd), newbond);

        Cs_[abInd[1]][0] *= delta(comInd, dag(newbond));
        Cs_[abInd[1]][1] *= delta(comInd, dag(newbond));
        Ts_[abInd[1]][1] *= delta(comInd, dag(newbond));
        Ts_[abInd[1]][3] *= delta(comInd, dag(newbond));
    } else {
        auto comInd = commonIndex(Cs_[abInd[0]][0], Cs_[abInd[1]][1]);
        auto newbond = cpInd(comInd, {"IndexName", "env", "IndexType", Wtype});
        Cs_[abInd[0]][0] *= delta(dag(comInd), newbond);
        Cs_[abInd[0]][3] *= delta(dag(comInd), newbond);
        Ts_[abInd[0]][0] *= delta(dag(comInd), newbond);
        Ts_[abInd[0]][2] *= delta(dag(comInd), newbond);

        Cs_[abInd[1]][1] *= delta(comInd, dag(newbond));
        Cs_[abInd[1]][2] *= delta(comInd, dag(newbond));
        Ts_[abInd[1]][0] *= delta(comInd, dag(newbond));
        Ts_[abInd[1]][2] *= delta(comInd, dag(newbond));
    }

}

// construct ctm tensors from the bulk tensors by constucting the reduncdant indies.
template<class IndexT>
void CTM<IndexT>::EnvFromAs() {
    auto traceBond = [this](TensorT &T, const int dir, const int bond) {
        T *= ((dir == 0) || (dir == 3)) ? dag(combAB_.at(bond))
                                        : combAB_.at(bond);
        T *= ((dir == 0) || (dir == 3)) ?
             delta(dag(links_.at(bond)), prime(links_.at(bond)))
                                        : delta(links_.at(bond), dag(prime(links_.at(bond))));
    };
    Cs_.clear();
    Ts_.clear();
    for (auto si:range(nSite_)) {
        array<TensorT, 4> Ctmp, Ttmp;
        auto AA = AAs_[si];
        auto sB = siteBonds(si, nx_, ny_, unitCell_);

        // c1,c2,c3,c4
        for (auto j:range(4)) {
            auto AAt = AA;
            traceBond(AAt, (-j + 1 + 4) % 4, sB[(-j + 1 + 4) % 4]);
            traceBond(AAt, (-j + 4) % 4, sB[(-j + 4) % 4]);
            Ctmp[j] = AAt;
        }
        // T1,T2,T3,T4
        for (auto j:range(4)) {
            auto AAt = AA;
            traceBond(AAt, (-j + 4) % 4, sB[(-j + 4) % 4]);
            Ttmp[j] = AAt;
        }
        Cs_.push_back(Ctmp);
        Ts_.push_back(Ttmp);
    }
    for (auto bond:range(2 * nSite_))
        resetEnvLink(bond);

    scaleEnv();
}

template<class IndexT>
void CTM<IndexT>::scaleAs() {
    for (auto &I:As_) {
        scaleTensor(I);
    }

    updatedEnv_ = -1;
}

template<class IndexT>
void CTM<IndexT>::scaleEnv() {
    for (auto &ICS:Cs_) {
        for (auto &ics:ICS) {
            scaleTensor(ics);
        }
    }
    for (auto &ITS:Ts_) {
        for (auto &its:ITS) {
            scaleTensor(its);
        }
    }

    updatedEnv_ = -1;
}

template<class IndexT>
void CTM<IndexT>::randEnv() {
    for (auto &ICS:Cs_) {
        for (auto &ics:ICS) {
            randomFill(ics);
        }
    }
    for (auto &ITS:Ts_) {
        for (auto &its:ITS) {
            randomFill(its);
        }
    }

    updatedEnv_ = -1;
}

//combine tensor T at si. except bond.
// 0<=si<nSite_;
// 0<= bond <= 2 * nSite_
template<class IndexT>
void CTM<IndexT>::combineBond(TensorT &T, const int si, const int bond) {
    auto bondarray = siteBonds(si, nx_, ny_, unitCell_);
    for (auto i:range(4)) {
        if (bond == bondarray[i]) continue;
        T *= ((i == 0) || (i == 3)) ? combAB_.at(bondarray[i])
                                    : dag(combAB_.at(bondarray[i]));
    }
}

//extract tensor T at si. except bond.
// 0<=si<nSite_;
// 0<= bond <= 2 * nSite_
template<class IndexT>
void CTM<IndexT>::extractBond(TensorT &T, const int si, const int bond) {
    auto bondarray = siteBonds(si, nx_, ny_, unitCell_);
    for (auto i:range(4)) {
        if (bond == bondarray[i]) continue;
        T *= ((i == 0) || (i == 3)) ? dag(combAB_.at(bondarray[i]))
                                    : combAB_.at(bondarray[i]);
    }
}

// construct bulk and CTM tensor with the left up tensor being site si
// 0<= si <= nx_*ny_
template<class IndexT>
void CTM<IndexT>::constructBulkEnv(const int si) {
    if (si < 0 || si >= (nx_ * ny_))
        cout << "can only return env with 0<=site< nx*ny" << endl;
    int line = si / nx_;
    int col = si % nx_;
    bulkT_.clear();
    bulkT_.push_back(unitCell_[indLC(line, col, nx_, ny_)]);
    bulkT_.push_back(unitCell_[indLC(line, col + 1, nx_, ny_)]);
    bulkT_.push_back(unitCell_[indLC(line + 1, col, nx_, ny_)]);
    bulkT_.push_back(unitCell_[indLC(line + 1, col + 1, nx_, ny_)]);

    envT_.clear();
    envT_.push_back(array<int, 2>{{unitCell_[indLC(line - 1, col - 1, nx_, ny_)], 0}});
    envT_.push_back(array<int, 2>{{unitCell_[indLC(line - 1, col + 0, nx_, ny_)], 0}});
    envT_.push_back(array<int, 2>{{unitCell_[indLC(line - 1, col + 1, nx_, ny_)], 0}});

    envT_.push_back(array<int, 2>{{unitCell_[indLC(line - 1, col + 2, nx_, ny_)], 1}});
    envT_.push_back(array<int, 2>{{unitCell_[indLC(line + 0, col + 2, nx_, ny_)], 1}});
    envT_.push_back(array<int, 2>{{unitCell_[indLC(line + 1, col + 2, nx_, ny_)], 1}});

    envT_.push_back(array<int, 2>{{unitCell_[indLC(line + 2, col + 2, nx_, ny_)], 2}});
    envT_.push_back(array<int, 2>{{unitCell_[indLC(line + 2, col + 1, nx_, ny_)], 2}});
    envT_.push_back(array<int, 2>{{unitCell_[indLC(line + 2, col + 0, nx_, ny_)], 2}});

    envT_.push_back(array<int, 2>{{unitCell_[indLC(line + 2, col - 1, nx_, ny_)], 3}});
    envT_.push_back(array<int, 2>{{unitCell_[indLC(line + 1, col - 1, nx_, ny_)], 3}});
    envT_.push_back(array<int, 2>{{unitCell_[indLC(line + 0, col - 1, nx_, ny_)], 3}});

    constructed_ = si;
}

template<class IndexT>
array<ITensorT<IndexT>, 2>
CTM<IndexT>::projOP(const vector<TensorT> &Qs, const array<IndexT, 2> Qind, const array<IndexT, 2> &Rind) {
//    clock_t tbegin,tend;
//    tbegin=std::clock();

    TensorT tQ1 = Qs[0];
    TensorT tQ2 = Qs[1];
    TensorT tQ3 = Qs[2];
    TensorT tQ4 = Qs[3];

    tQ2.prime(Qind[0], Qind[1]);
    tQ4.prime(Qind[0], Qind[1]);
    auto upHalf = tQ1 * tQ2;
    auto downHalf = tQ3 * tQ4;

//    tend=std::clock();
//    cout<<"build UP Down tensor time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;

//    tbegin=std::clock();
    auto R = TensorT(Rind[0], Rind[1]);
    auto tildeR = TensorT(dag(Rind[0]), dag(Rind[1]));
    TensorT Q, tildeQ;
    QRdecomp(upHalf, Q, R, false);
    QRdecomp(downHalf, tildeQ, tildeR, false);
//    tend=std::clock();
//    cout<<"QR time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
//    auto U = TensorT(commonIndex(R,Q));
    auto U = TensorT(uniqueIndex(R, upHalf, All));
    TensorT A, V, S;
    A = R * tildeR;
//    tbegin=std::clock();
    auto spec = svd(A, U, S, V, {"Maxm", chi_, "LeftIndexType", Wtype, "RightIndexType", Wtype});
//    tend=std::clock();
//    cout<<"SVD time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
//    Print(spec.truncerr());

//    Print(U);
//    Print(S);
//    tbegin=std::clock();
    S.scaleTo(1.0);
    S.apply([](Real x) { return 1.0 / sqrt(x); });
    auto tildeP = tildeR * dag(V) * dag(S);
    auto P = dag(S) * dag(U) * R;
    P *= delta(commonIndex(S, U), commonIndex(S, V));
//    tend=std::clock();
//    cout<<"build p time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
    return array<TensorT, 2>{{P, tildeP}};
}

// inset one column labeled with "col" to the left.
// 0<= col< nx_;
template<class IndexT>
vector<ITensorT<IndexT> > CTM<IndexT>::leftCTMRGStep(const int col, const bool his) {
    vector<TensorT> tildePs;
    vector<TensorT> Ps;
    vector<TensorT> bulkTs;
    for (auto line:range(ny_)) {
        auto ind = indLC(line, col, nx_, ny_);
        auto env = returnEnv(ind);
        auto bulk = returnBulk(ind);
        if (his) {
            if (bulkT_[0] == abInd_[0]) {
                bulk[0] = AA_;
                bulk[1] = BB_;
            }
            if (bulkT_[2] == abInd_[0]) {
                bulk[2] = AA_;
                bulk[3] = BB_;
            }
        }
        bulkTs.push_back(bulk[0]);
        vector<TensorT> Qs;
        Qs.push_back(env[0] * env[1] * env[11] * bulk[0]);
        Qs.push_back(env[3] * env[2] * env[4] * bulk[1]);
        Qs.push_back(env[9] * env[8] * env[10] * bulk[2]);
        Qs.push_back(env[6] * env[5] * env[7] * bulk[3]);

        array<IndexT, 2> RindV, QindV;
        RindV[0] = commonUniq(env[11], env[10], env[9]);
        auto sBR = siteBonds(indLC(line, col, nx_, ny_), nx_, ny_, unitCell_);
        RindV[1] = commonIndex(bulk[0], combAB_[sBR[2]]);

        QindV[0] = commonUniq(env[4], env[5], env[6]);
        auto sBQ = siteBonds(indLC(line, col + 1, nx_, ny_), nx_, ny_, unitCell_);
        QindV[1] = commonIndex(bulk[1], combAB_[sBQ[2]]);

        auto projs = projOP(Qs, QindV, RindV);

        Ps.push_back(projs[0]);
        tildePs.push_back(projs[1]);
    }

    vector<TensorT> newEnvs;
    for (auto line:range(ny_)) {
        auto ind0 = unitCell_[indLC(line, col - 1, nx_, ny_)];
        auto ind1 = unitCell_[indLC(line, col, nx_, ny_)];

        newEnvs.push_back(Cs_[ind0][0] * Ts_[ind1][0]
                          * tildePs[line]);
        newEnvs.push_back(Ps[(line + ny_ - 1) % ny_]
                          * Ts_[ind0][3] * bulkTs[line]
                          * tildePs[line]);
        newEnvs.push_back(Cs_[ind0][3] * Ts_[ind1][2]
                          * Ps[(line + ny_ - 1) % ny_]);
    }

    for (auto &T:newEnvs)
        scaleTensor(T);
    updatedEnv_ = -1;

    return newEnvs;
}

// inset one column labeled with "col" to the right.
// 0<= col< nx_;
template<class IndexT>
vector<ITensorT<IndexT> > CTM<IndexT>::rightCTMRGStep(const int col, const bool his) {
    vector<TensorT> tildePs;
    vector<TensorT> Ps;
    vector<TensorT> bulkTs;
    for (auto line:range(ny_)) {
        auto ind = indLC(line, col - 1, nx_, ny_);
        auto env = returnEnv(ind);
        auto bulk = returnBulk(ind);
        if (his) {
            if (bulkT_[0] == abInd_[0]) {
                bulk[0] = AA_;
                bulk[1] = BB_;
            }
            if (bulkT_[2] == abInd_[0]) {
                bulk[2] = AA_;
                bulk[3] = BB_;
            }
        }
        bulkTs.push_back(bulk[1]);
        vector<TensorT> Qs;
        Qs.push_back(env[3] * env[2] * env[4] * bulk[1]);
        Qs.push_back(env[0] * env[1] * env[11] * bulk[0]);
        Qs.push_back(env[6] * env[5] * env[7] * bulk[3]);
        Qs.push_back(env[9] * env[8] * env[10] * bulk[2]);

        array<IndexT, 2> RindV, QindV;
        RindV[0] = commonUniq(env[4], env[5], env[6]);
        auto sBR = siteBonds(indLC(line, col, nx_, ny_), nx_, ny_, unitCell_);
        RindV[1] = commonIndex(bulk[1], combAB_[sBR[2]]);

        QindV[0] = commonUniq(env[11], env[10], env[9]);
        auto sBQ = siteBonds(indLC(line, col - 1, nx_, ny_), nx_, ny_, unitCell_);
        QindV[1] = commonIndex(bulk[0], combAB_[sBQ[2]]);

        auto projs = projOP(Qs, QindV, RindV);
        Ps.push_back(projs[0]);
        tildePs.push_back(projs[1]);
    }
    vector<TensorT> newEnvs;
    for (auto line:range(ny_)) {
        auto ind0 = unitCell_[indLC(line, col + 1, nx_, ny_)];
        auto ind1 = unitCell_[indLC(line, col, nx_, ny_)];
        newEnvs.push_back(Cs_[ind0][1] * Ts_[ind1][0]
                          * tildePs[line]);
        newEnvs.push_back(Ps[(line + ny_ - 1) % ny_]
                          * Ts_[ind0][1] * bulkTs[line]
                          * tildePs[line]);
        newEnvs.push_back(Cs_[ind0][2] * Ts_[ind1][2]
                          * Ps[(line + ny_ - 1) % ny_]);
    }
    for (auto &T:newEnvs)
        scaleTensor(T);
    updatedEnv_ = -1;
    return newEnvs;

}

// inset one row labeled with "line" to the up.
// 0<= line< ny_;
template<class IndexT>
vector<ITensorT<IndexT> > CTM<IndexT>::upCTMRGStep(const int line, const bool his) {
    vector<TensorT> tildePs;
    vector<TensorT> Ps;
    vector<TensorT> bulkTs;
    for (auto col:range(nx_)) {
        auto ind = indLC(line, col, nx_, ny_);
        auto env = returnEnv(ind);
        auto bulk = returnBulk(ind);
        if (his) {
            if (bulkT_[0] == abInd_[1]) {
                bulk[2] = AA_;
                bulk[0] = BB_;
            }
            if (bulkT_[1] == abInd_[1]) {
                bulk[3] = AA_;
                bulk[1] = BB_;
            }
        }
        bulkTs.push_back(bulk[0]);

        vector<TensorT> Qs;
        Qs.push_back(env[0] * env[1] * env[11] * bulk[0]);
        Qs.push_back(env[9] * env[8] * env[10] * bulk[2]);
        Qs.push_back(env[3] * env[2] * env[4] * bulk[1]);
        Qs.push_back(env[6] * env[5] * env[7] * bulk[3]);

        array<IndexT, 2> RindV, QindV;
        RindV[0] = commonUniq(env[1], env[2], env[3]);;
        auto sBR = siteBonds(indLC(line, col, nx_, ny_), nx_, ny_, unitCell_);
        RindV[1] = commonIndex(bulk[0], combAB_[sBR[3]]);

        QindV[0] = commonUniq(env[8], env[7], env[6]);
        auto sBQ = siteBonds(indLC(line + 1, col, nx_, ny_), nx_, ny_, unitCell_);
        QindV[1] = commonIndex(bulk[2], combAB_[sBQ[3]]);

        auto projs = projOP(Qs, QindV, RindV);
        Ps.push_back(projs[0]);
        tildePs.push_back(projs[1]);
    }

    vector<TensorT> newEnvs;
    for (auto col:range(nx_)) {
        auto ind0 = unitCell_[indLC(line - 1, col, nx_, ny_)];
        auto ind1 = unitCell_[indLC(line, col, nx_, ny_)];
        newEnvs.push_back(Cs_[ind0][0] * Ts_[ind1][3]
                          * tildePs[col]);
        newEnvs.push_back(Ps[(col + nx_ - 1) % nx_]
                          * Ts_[ind0][0] * bulkTs[col]
                          * tildePs[col]);
        newEnvs.push_back(Cs_[ind0][1] * Ts_[ind1][1]
                          * Ps[(col + nx_ - 1) % nx_]);
    }

    for (auto &T:newEnvs)
        scaleTensor(T);
    updatedEnv_ = -1;
    return newEnvs;
}

// inset one row labeled with "line" to the buttom.
// 0<= line< ny_;
template<class IndexT>
vector<ITensorT<IndexT> > CTM<IndexT>::downCTMRGStep(const int line, const bool his) {
    vector<TensorT> tildePs;
    vector<TensorT> Ps;
    vector<TensorT> bulkTs;
//    clock_t tbegin,tend;
//    tbegin=std::clock();
    for (auto col:range(nx_)) {
        auto ind = indLC(line - 1, col, nx_, ny_);
        auto env = returnEnv(ind);
        auto bulk = returnBulk(ind);
        if (his) {
            if (bulkT_[0] == abInd_[1]) {
                bulk[2] = AA_;
                bulk[0] = BB_;
            }
            if (bulkT_[1] == abInd_[1]) {
                bulk[3] = AA_;
                bulk[1] = BB_;
            }
        }
        bulkTs.push_back(bulk[2]);

//        tbegin=std::clock();
        vector<TensorT> Qs;
        Qs.push_back(env[9] * env[8] * env[10] * bulk[2]);
        Qs.push_back(env[0] * env[1] * env[11] * bulk[0]);
        Qs.push_back(env[6] * env[5] * env[7] * bulk[3]);
        Qs.push_back(env[3] * env[2] * env[4] * bulk[1]);
//        tend=std::clock();
//        cout<<"Build Qs time:    "
//            <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;

        array<IndexT, 2> RindV, QindV;
        RindV[0] = commonUniq(env[8], env[7], env[6]);;
        auto sBR = siteBonds(indLC(line, col, nx_, ny_), nx_, ny_, unitCell_);
        RindV[1] = commonIndex(bulk[2], combAB_[sBR[3]]);

        QindV[0] = commonUniq(env[1], env[2], env[3]);
        auto sBQ = siteBonds(indLC(line - 1, col, nx_, ny_), nx_, ny_, unitCell_);
        QindV[1] = commonIndex(bulk[0], combAB_[sBQ[3]]);

//        tbegin=std::clock();
        auto projs = projOP(Qs, QindV, RindV);
//        tend=std::clock();
//        cout<<"Build pop time:    "
//            <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
        Ps.push_back(projs[0]);
        tildePs.push_back(projs[1]);
    }
//    tend=std::clock();
//    cout<<"build project operator time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;

//    tbegin=std::clock();
    vector<TensorT> newEnvs;
    for (auto col:range(nx_)) {
        auto ind0 = unitCell_[indLC(line + 1, col, nx_, ny_)];
        auto ind1 = unitCell_[indLC(line, col, nx_, ny_)];
        newEnvs.push_back(Cs_[ind0][3] * Ts_[ind1][3]
                          * tildePs[col]);
        newEnvs.push_back(Ps[(col + nx_ - 1) % nx_]
                          * Ts_[ind0][2] * bulkTs[col]
                          * tildePs[col]);
        newEnvs.push_back(Cs_[ind0][2] * Ts_[ind1][1]
                          * Ps[(col + nx_ - 1) % nx_]);
    }

//    tend=std::clock();
//    cout<<"New env time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;

//    tbegin=std::clock();
    for (auto &T:newEnvs)
        scaleTensor(T);
//    tend=std::clock();
//    cout<<"scale time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
    updatedEnv_ = -1;
    return newEnvs;
}

template<class IndexT>
void CTM<IndexT>::CTMRGstep() {
//    clock_t tbegin,tend;
    for (auto col:range(nx_)) {
//        cout << "left inserting col:" << col << endl;
        auto newEnvs = leftCTMRGStep(col);

        for (auto line:range(ny_)) {
            auto ind1 = unitCell_[indLC(line, col, nx_, ny_)];
            Cs_[ind1][0] = newEnvs[3 * line + 0];
            Ts_[ind1][3] = newEnvs[3 * line + 1];
            Cs_[ind1][3] = newEnvs[3 * line + 2];
        }
    }

    for (auto col:range(nx_)) {
//        cout << "right inserting col:" << nx_-col -1 << endl;
        auto newEnvs = rightCTMRGStep(nx_ - col - 1);
        for (auto line:range(ny_)) {
            auto ind1 = unitCell_[indLC(line, nx_ - col - 1, nx_, ny_)];
            Cs_[ind1][1] = newEnvs[3 * line + 0];
            Ts_[ind1][1] = newEnvs[3 * line + 1];
            Cs_[ind1][2] = newEnvs[3 * line + 2];
        }
    }

    for (auto line:range(ny_)) {
//        cout << "up inserting line:" << line << endl;
        auto newEnvs = upCTMRGStep(line);

        for (auto col:range(nx_)) {
            auto ind1 = unitCell_[indLC(line, col, nx_, ny_)];
            Cs_[ind1][0] = newEnvs[3 * col + 0];
            Ts_[ind1][0] = newEnvs[3 * col + 1];
            Cs_[ind1][1] = newEnvs[3 * col + 2];
        }
    }

    for (auto line:range(ny_)) {
//        cout << "down inserting line:" << ny_ - line -1 << endl;
        auto newEnvs = downCTMRGStep(ny_ - line - 1);

//        tbegin=std::clock();
        for (auto col:range(nx_)) {
            auto ind1 = unitCell_[indLC(ny_ - line - 1, col, nx_, ny_)];
            Cs_[ind1][3] = newEnvs[3 * col + 0];
            Ts_[ind1][2] = newEnvs[3 * col + 1];
            Cs_[ind1][2] = newEnvs[3 * col + 2];
        }
//        tend=std::clock();
//        cout<<"Update env time:    "
//            <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
    }

    updatedEnv_ = -1;
}

template<class IndexT>
Real CTM<IndexT>::CTMRG(const int maxRGStep, const Real error) {
//    clock_t tbegin,tend;

    auto esold = cornerES();
    int numRG = 0;
    Real des = 1.0, desold = 0.5;
    for (numRG = 0; numRG < maxRGStep; ++numRG) {
//        tbegin=std::clock();
        CTMRGstep();
        checkNorm();
//        tend=std::clock();
//        cout<<"RG step time:    "
//            <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
//        tbegin=std::clock();
        auto esnew = cornerES();
        vector<Real> desV;
        for (auto i:range(esnew.size()))
            desV.push_back(abs(esnew.at(i) - esold.at(i)));
        auto desp = max_element(desV.begin(), desV.end());
        des = *desp;
        cout << "CTMRG step " << numRG + 1 << " with relevant error " << des << endl;
//        if ( ( numRG>10&& abs(des)>abs(desold) )|| abs(des)< error) {
        if (abs(des) < error) {
            cout << "CTM converged at RG step:" << numRG + 1
                 << ", with relevant error:" << des << endl;
            break;
        }
//        tend=std::clock();
//        cout<<"estimate error time:    "
//            <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
        esold = esnew;
        desold = des;
    }
    if (numRG >= maxRGStep - 1)
        cout << "Maximun CTMRG step reached"
             << ", with relevant error:" << des << endl;

    updatedEnv_ = -1;

    return des;
}

template<class IndexT>
vector<Real> CTM<IndexT>::cornerES() const {
    vector<Real> ES;
    TensorT D, V;
    for (auto &tC:Cs_) {
        for (auto i:range(4)) {
            auto U = TensorT(tC[i].inds().front());
            auto sp = svd(tC[i] / norm(tC[i]), U, D, V);
            ES.push_back(entropy(sp));
        }
    }
    return ES;
}

// 0<= bond <= 2 * nSite_
template<class IndexT>
void CTM<IndexT>::reduceAB(TensorT &AX, TensorT &BY, TensorT &aR, TensorT &bL,
                           TensorT &AAX, TensorT &BBY, const int bond) {
    auto vert = (bond % 2 == 0);
    auto indAB = calcABInd(bond, nx_, ny_, unitCell_);
    auto linkAarray = siteBonds(indAB[0], nx_, ny_, unitCell_);
    auto linkBarray = siteBonds(indAB[1], nx_, ny_, unitCell_);

    auto A = As_[indAB[0]];
    auto B = As_[indAB[1]];
    if (isFerm_) {
        if (vert) {
            if (indAB[0] == 0) fSwapGate(A, symInd_, findtype(A, Site));
            fSwapGate(A, links_[linkAarray[1]], findtype(A, Site));
        } else {
            fSwapGate(A, links_[linkAarray[2]], findtype(A, Site));
            if (indAB[1] == 0) fSwapGate(B, symInd_, findtype(B, Site));
        }
    }
    TensorT AX1, BY1, U;
    aR = TensorT(findtype(A, Site), links_[bond]);
    bL = TensorT(findtype(B, Site), links_[bond]);
//    denmatDecomp(A_,AX1,aR,Fromleft,
//                 {"IndexName", "aa","Truncate",false});
//    denmatDecomp(B_,BY1,bL,Fromleft,
//                 {"IndexName", "bb","Truncate",false});
    QRdecomp(A, AX1, aR);
    QRdecomp(B, BY1, bL);
    auto aind = commonIndex(AX1, aR);
    auto bind = commonIndex(BY1, bL);
    AX = AX1;
    BY = BY1;
    auto barAX = dag(AX);
    auto barBY = dag(BY);

    if (isFerm_) {
        if (vert) {
            fSwapGate(barAX, aind, links_[linkAarray[1]]);
            fSwapGate(barAX, links_[linkAarray[2]], links_[linkAarray[3]]);

            fSwapGate(barBY, links_[linkBarray[0]], links_[linkBarray[1]]);
            fSwapGate(barBY, bind, links_[linkBarray[3]]);
        } else {
            fSwapGate(barAX, links_[linkAarray[0]], links_[linkAarray[1]]);
            fSwapGate(barAX, links_[linkAarray[2]], aind);

            fSwapGate(barBY, links_[linkBarray[0]], bind);
            fSwapGate(barBY, links_[linkBarray[2]], links_[linkBarray[3]]);
        }
    }

    AAX = AX * primeExcept(barAX, dag(symInd_));
    BBY = BY * primeExcept(barBY, dag(symInd_));

    if (isFerm_) {
        if (vert) {
            fSwapGate(AAX, prime(aind), links_[linkAarray[1]]);
            fSwapGate(AAX, links_[linkAarray[2]], prime(links_[linkAarray[3]]));

            fSwapGate(BBY, prime(links_[linkBarray[0]]), links_[linkBarray[1]]);
            fSwapGate(BBY, bind, prime(links_[linkBarray[3]]));
        } else {
            fSwapGate(AAX, prime(links_[linkAarray[0]]), links_[linkAarray[1]]);
            fSwapGate(AAX, links_[linkAarray[2]], prime(aind));

            fSwapGate(BBY, prime(links_[linkBarray[0]]), bind);
            fSwapGate(BBY, links_[linkBarray[2]], prime(links_[linkBarray[3]]));
        }
    }

    combineBond(AAX, indAB[0], bond);
    combineBond(BBY, indAB[1], bond);
}

// construct the bond env tensor.
// 0<= bond <= 2 * nSite_
template<class IndexT>
void CTM<IndexT>::bondEnv(const int bond) {
    auto vert = (bond % 2 == 0);
    abInd_ = calcABInd(bond, nx_, ny_, unitCell_);
    auto ind = vert ? abInd_[1] : abInd_[0];
    auto env = returnEnv(ind);
    auto bulk = returnBulk(ind);

    TensorT AAX, BBY;
    reduceAB(AX_, BY_, aR_, bL_, AAX, BBY, bond);

//    clock_t tbegin, tend;
//    tbegin=clock();
    auto Q1 = env[0] * env[1] * env[11];
    auto Q2 = env[3] * env[2] * env[4];
    auto Q3 = env[6] * env[5] * env[7];
    auto Q4 = env[9] * env[8] * env[10];

    if (vert) {
        Q1 *= BBY;
        Q2 *= bulk[1];
        Q3 *= bulk[3];
        Q4 *= AAX;
        bondenv_ = Q1 * Q2 * Q3 * Q4;
    } else {
        Q1 *= AAX;
        Q2 *= BBY;
        Q3 *= bulk[3];
        Q4 *= bulk[2];
        bondenv_ = Q1 * Q4 * Q3 * Q2;
    }
//    tend=std::clock();
//    cout<<"Build bondenv time:    "
//         <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;

//    bondenv_ = Q1*Q2*Q3*Q4;

    auto bondTensor = aR_ * bL_
                      * primeExcept(dag(aR_) * dag(bL_), Site);

    norm_ = bondenv_ * bondTensor;
    updatedEnv_ = bond;
}

// expected value of one site operator op at site si.
// 0<= si <= nSite_
template<class IndexT>
Real CTM<IndexT>::exp_site(const TensorT &op, const int si) {
    auto bondarray = siteBonds(si, nx_, ny_, unitCell_);
    auto bond = bondarray[3];
    if (updatedEnv_ != bond) bondEnv(bond);
    auto bondTensor = aR_ * op * prime(dag(aR_));
    bondTensor *= bL_ * primeExcept(dag(bL_), Site);

    auto exp = bondenv_ * bondTensor;

    return exp.real() / norm_.real();
}

// expected value of two site operator op at bond.
// 0<= bond <= 2* nSite_
template<class IndexT>
Real CTM<IndexT>::exp_bond(const TensorT &op, const int bond) {
    if (updatedEnv_ != bond) bondEnv(bond);
    auto bondTensor = aR_ * bL_ * op;
    bondTensor *= prime(dag(aR_) * dag(bL_));

    auto exp = bondenv_ * bondTensor;

    return exp.real() / norm_.real();
}

template<class IndexT>
void CTM<IndexT>::fixGauge(const int bond, TensorT &tildeAR, TensorT &tildeBL,
                           TensorT &tildeN, TensorT &invL, TensorT &invR) {
    if (updatedEnv_ != bond) bondEnv(bond);
    auto sym = norm(bondenv_ - dag(swapPrime(bondenv_, 0, 1)));
    if (sym > 1.0e-12)
        cout << "Enviroment non-Hermite ~ " << sym << ", Symmetrize needed!" << endl;
    bondenv_ += dag(swapPrime(bondenv_, 0, 1));
    bondenv_ /= 2.0;
    TensorT W, D;
    diagHermitian(bondenv_, W, D);
//    cout<<"before truncation"<<endl;
//    PrintDat(D);
    D.scaleTo(1.0);
    Real elesum = 0.0;
    Real minele = 1000.0;
    Real maxele = -1000.0;
    Real minAbs = 1000;
    int numneg = 0;
    int tnum = 0;
    auto accum = [&elesum, &minele, &maxele, &minAbs, &numneg, &tnum](Real r) {
        if (r < 0.0) numneg++;
        if (r < minele) minele = r;
        if (r > maxele) maxele = r;
        if (abs(r) < minAbs) minAbs = abs(r);
        elesum += r;
        tnum++;
    };
    D.visit(accum);
    cout << " info of env tensor" << endl;
    cout << "min" << " ~~ " << "max" << "  " << "min of Abs" << "  " << "sum" << " , " << "number of negtive egs "
         << " / " << "total egs" << endl;
    cout << minele << " ~~  " << maxele << "  " << minAbs << "  " << elesum << " ,  " << numneg << " / " << tnum
         << endl;

    bondenv_ /= maxAbs(D);
    if (elesum < 0.0) {
        cout << "warning!! negetive env" << endl;
        bondenv_ *= -1.0;

//        for(auto i:range(nSite_))
//            writeToFile(format("critical_As_%d", i), As_.at(i));

//        for(auto i:range(2*nSite_))
//            writeToFile(format("critical_Links_%d", i), links_.at(i));

    }
    diagHermitian(bondenv_, W, D, {"Cutoff", 1.0e-14});
//    cout<<"After truncation"<<endl;
//    PrintDat(D);
    auto cind = commonIndex(W, D);
    D.scaleTo(1.0);
    D.apply([](Real x) { return sqrt(x); });
    auto Z = noprime(dag(W) * D);
//    Print(norm(bondenv_-Z*primeExcept(dag(Z),cind)));

    auto lind = commonIndex(Z, aR_);
    auto rind = commonIndex(Z, bL_);
    TensorT qL, qR, R, L;
    R = TensorT(rind);
    L = TensorT(lind);
    QRdecomp(Z, qL, R);
    QRdecomp(Z, qR, L);

    matInverse(R, invR);
    matInverse(L, invL);

    tildeAR = aR_ * L;
    tildeBL = bL_ * R;
    auto tildeZ = invL * Z * invR;
    tildeN = tildeZ * primeExcept(dag(tildeZ), cind);
}

// apply gate at bond and updated iteratertively.
// 0<= bond <= 2* nSite_
template<class IndexT>
Real CTM<IndexT>::updateAB(const TensorT &gate, const int bond, IndexT &cind,
                           const int maxloop, const Real error) {
    if (updatedEnv_ != bond) bondEnv(bond);
    cout << "state normalization:" << norm_.real() << endl;
    // update A

    auto vert = (bond % 2 == 0);

    TensorT tildeAR, tildeBL, tildeN, invL, invR;
    fixGauge(bond, tildeAR, tildeBL, tildeN, invL, invR);
    auto tEvBond = noprime(tildeAR * tildeBL * gate);
    auto bondtensor = tEvBond * primeExcept(dag(tEvBond), Site);
    //calculate tensorT
    auto tensorT = tildeN * bondtensor;

    auto aR = tildeAR;
    auto bL = tildeBL;

//initiate A and B
//    factor(tEvBond,aR,bL,{"Maxm", bondD_, "IndexName", "ab"});
    denmatDecomp(tEvBond, aR, bL, Fromright, {"Maxm", bondD_, "IndexName", "ab"});

//    aR *= qDelta(dag(cind),links_[bond]);
//    bL *= qDelta(cind,dag(links_[bond]));

    Real dold = 1.0;
    Real dnew = 0.0;
    int loop = 0;
    for (loop = 0; loop < maxloop; ++loop) {
        TensorT lPoint, rPoint;
        for (auto ab:range(2)) {
            lPoint = (ab == 0) ? aR : bL;
            rPoint = (ab == 0) ? bL : aR;
            bool fromA = (ab == 0) ? false : true;
            bondtensor = tEvBond * primeExcept(dag(rPoint), Site);
            auto tensorS = tildeN * bondtensor;
            bondtensor = (rPoint) * primeExcept(dag(rPoint), Site);
            auto tensorR = tildeN * bondtensor;
            auto normS = maxAbs(tensorS);
            tensorR /= normS;
            tensorS /= normS;
            tensorR.scaleTo(1.0);
            tensorS.scaleTo(1.0);
            tensorR += dag(swapPrime(tensorR, 0, 1));
            tensorR /= 2.0;
            solveLinearEq(tensorR, tensorS, lPoint);

            dnew = tensorT.real() / normS
                   + (primeExcept(dag(lPoint), Site) * tensorR * lPoint).real()
                   - (primeExcept(dag(lPoint), Site) * tensorS).real()
                   - (noprime(dag(tensorS)) * lPoint).real();
            dnew *= normS;
            dnew /= tensorT.real();

            auto rl = TensorT(commonIndex(lPoint, rPoint));
            TensorT ql;
            QRdecomp(lPoint, ql, rl);
            if (ab == 0) {
                aR = ql;
                bL = rl * rPoint;
            } else {
                bL = ql;
                aR = rl * rPoint;
            }
        }
        cout << "full update step " << loop + 1 << " with T= "
             << tensorT.real() << " and relevant error " << dnew << endl;
        if (abs(dnew - dold) < error) {
            cout << "full update converged at loop " << loop + 1 << " with T= "
                 << tensorT.real() << " and relative error " << dnew << endl;
            break;
        }
        dold = dnew;
    }
    if (loop >= maxloop - 1)
        cout << "maximun full update iteration reached,"
             << " with T= " << tensorT.real() << ", and relative error " << dnew << endl;

    cind = commonIndex(aR, bL);
    TensorT qaR, raR, qbL, rbL;
    raR = TensorT(cind);
    rbL = TensorT(cind);
    QRdecomp(aR, qaR, raR);
    QRdecomp(bL, qbL, rbL);
    factor(raR * rbL, raR, rbL);
    aR = invL * qaR * raR;
    bL = invR * qbL * rbL;
    A_ = AX_ * aR;
    B_ = BY_ * bL;
    scaleTensor(A_);
    scaleTensor(B_);
    cind = commonIndex(raR, rbL);

    auto indAB = calcABInd(bond, nx_, ny_, unitCell_);
    auto linkAarray = siteBonds(indAB[0], nx_, ny_, unitCell_);
    if (isFerm_) {
        if (vert) {
            fSwapGate(A_, links_[linkAarray[1]], findtype(A_, Site));
            if (indAB[0] == 0) fSwapGate(A_, symInd_, findtype(A_, Site));
        } else {
            fSwapGate(A_, links_[linkAarray[2]], findtype(A_, Site));
            if (indAB[1] == 0) fSwapGate(B_, symInd_, findtype(B_, Site));
        }
    }

    tEvBond = noprime(aR_ * bL_ * gate);
    bondtensor = tEvBond * primeExcept(dag(tEvBond), Site);
    auto pp = (bondenv_ * bondtensor).real();

    bondtensor = tEvBond * primeExcept(dag(aR) * dag(bL), Site);
    auto pt = (bondenv_ * bondtensor).real();

    bondtensor = aR * bL * primeExcept(dag(aR) * dag(bL), Site);
    auto tt = (bondenv_ * bondtensor).real();

    dnew = sqrt((pp + tt - 2.0 * pt) / pp);

    cout << "Total full update error: " << dnew << endl;

    return dnew;
}

template<class IndexT>
Real CTM<IndexT>::reduceUpdate(const TensorT &gate, const int bond, const int maxloop, const Real error) {
    if (updatedEnv_ != bond) bondEnv(bond);
    IndexT cind;
    auto dFU = updateAB(gate, bond, cind, maxloop, error);

    auto changecind = isEqual(dag(cind), links_[bond]);
    if (changecind) {
        A_ *= qDelta(dag(cind), links_[bond]);
        B_ *= qDelta(cind, dag(links_[bond]));
    } else {
        cout << "Bond " << bond << " has been changed, reset env needed!" << endl;
        links_[bond] = cind;
        combAB_[bond] = combiner(cind, prime(dag(cind)));
    }

    AAfromA(AA_, A_, abInd_[0]);
    AAfromA(BB_, B_, abInd_[1]);

    As_[abInd_[0]] = A_;
    AAs_[abInd_[0]] = AA_;

    As_[abInd_[1]] = B_;
    AAs_[abInd_[1]] = BB_;

    if (!changecind) EnvFromAs();

    updatedEnv_ = -1;
    return dFU;
}

template<class IndexT>
void CTM<IndexT>::applySiteOp(const TensorT &op, const int si) {
    As_[si] *= op;
    As_[si].noprime(Site);
    scaleTensor(As_[si]);

    AAfromA(AA_, As_[si], si);
    AAs_[si] = AA_;

    updatedEnv_ = -1;
}

template<class IndexT>
Real CTM<IndexT>::fastUpdate(const TensorT &gate, const int bond, const int maxloop, const Real error) {
//    clock_t tbegin, tend;
//    tbegin=std::clock();
    if (updatedEnv_ != bond) bondEnv(bond);
    auto Aold = As_[abInd_[0]];
    auto Bold = As_[abInd_[1]];
    auto AAold = AAs_[abInd_[0]];
    auto BBold = AAs_[abInd_[1]];

    IndexT cind;
    auto dFU = updateAB(gate, bond, cind, maxloop, error);
//    tend=std::clock();
//    cout<<"Update AB time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
//    tbegin=std::clock();
    links_[bond] = cind;
    combAB_[bond] = combiner(cind, prime(dag(cind)));

    AAfromA(AA_, A_, abInd_[0]);
    AAfromA(BB_, B_, abInd_[1]);
//    tend=std::clock();
//    cout<<"Update Links time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
//    tbegin=clock();
    auto vert = (bond % 2 == 0);
    if (vert) {
        auto upLine = abInd_[1] / nx_;
        auto downLine = abInd_[0] / nx_;
        auto envUp = upCTMRGStep(upLine, true);
        auto envDown = downCTMRGStep(downLine, true);

        for (auto col:range(nx_)) {
            auto ind1 = unitCell_[indLC(upLine, col, nx_, ny_)];
            Cs_[ind1][0] = envUp[3 * col + 0];
            Ts_[ind1][0] = envUp[3 * col + 1];
            Cs_[ind1][1] = envUp[3 * col + 2];
        }
        for (auto col:range(nx_)) {
            auto ind1 = unitCell_[indLC(downLine, col, nx_, ny_)];
            Cs_[ind1][3] = envDown[3 * col + 0];
            Ts_[ind1][2] = envDown[3 * col + 1];
            Cs_[ind1][2] = envDown[3 * col + 2];
        }

    } else {
        auto leftCol = abInd_[0] % nx_;
        auto rightCol = abInd_[1] % nx_;
        auto leftenv = leftCTMRGStep(leftCol, true);
        auto rightenv = rightCTMRGStep(rightCol, true);
        for (auto line:range(ny_)) {
            auto ind1 = unitCell_[indLC(line, leftCol, nx_, ny_)];
            Cs_[ind1][0] = leftenv[3 * line + 0];
            Ts_[ind1][3] = leftenv[3 * line + 1];
            Cs_[ind1][3] = leftenv[3 * line + 2];
        }
        for (auto line:range(ny_)) {
            auto ind1 = unitCell_[indLC(line, rightCol, nx_, ny_)];
            Cs_[ind1][1] = rightenv[3 * line + 0];
            Ts_[ind1][1] = rightenv[3 * line + 1];
            Cs_[ind1][2] = rightenv[3 * line + 2];
        }
    }

//    tend=std::clock();
//    cout<<"CTM RG time:    "
//        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;

    As_[abInd_[0]] = A_;
    AAs_[abInd_[0]] = AA_;

    As_[abInd_[1]] = B_;
    AAs_[abInd_[1]] = BB_;

    return dFU;
}

template<class IndexT>
void CTM<IndexT>::drawEnvBulk(string name, bool print) {
    for (auto ni:range(nx_ * ny_)) {
//        cout<<"si: "<< ni<<endl;
        auto env = returnEnv(ni);
        auto bulk = returnBulk(ni);

        int line = ni / nx_;
        int col = ni % nx_;
        vector<IndexT> bulkLink;
        auto ind = indLC(line, col, nx_, ny_);
        auto sb = siteBonds(unitCell_[ind], nx_, ny_, unitCell_);
        bulkLink.push_back(dag(commonIndex(AAs_[unitCell_[ind]], combAB_[sb[2]])));
        bulkLink.push_back(dag(commonIndex(AAs_[unitCell_[ind]], combAB_[sb[3]])));
        ind = indLC(line + 1, col + 1, nx_, ny_);
        sb = siteBonds(unitCell_[ind], nx_, ny_, unitCell_);
        bulkLink.push_back(commonIndex(AAs_[unitCell_[ind]], combAB_[sb[0]]));
        bulkLink.push_back(commonIndex(AAs_[unitCell_[ind]], combAB_[sb[1]]));

        if (print) {
            cout << "########################" << endl;
            cout << "##### site (" << line << ", " << col << ") ####" << endl;
            for (auto &envT:env)
                Print(envT);
            for (auto &bulkT:bulk)
                Print(bulkT);
            cout << "   ................." << endl;
        }
        pstrickCTM<IndexT>(env, bulk, bulkLink, nameint(name, ni));
    }
}

template
class CTM<Index>;

template
class CTM<IQIndex>;