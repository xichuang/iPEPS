//
// Created by xich on 16-10-19.
//

#include "iPEPS.h"

/*
 * Initiating A, B, GA and LS with tensors given by user.
 */
template<class IndexT>
iPEPS<IndexT>::iPEPS(const vector<int> &uc, const SiteSet &sites,
                     vector<array<IndexT, 2> > &linksG, vector<TensorT> &Ls,
                     vector<TensorT> &G, IndexT &symInd,
                     const int nSite, int boundD, const int nx, const int ny,
                     const Args &args)
        :sites_(sites), nSite_(nSite),
         nx_(nx), ny_(ny),
         unitCell_(uc),
         dim_(boundD),
         linksG_(linksG),Ls_(Ls),G_(G),symInd_(symInd){
    isFerm_ = args.getBool("Fermionic",false);
}
/*
 * Initiating A, B, GA and LS with random tensors.
 */
template<>
iPEPS<Index>::iPEPS(const vector<int> &uc, const SiteSet &sites, const int nSite, int boundD, const int nx, const int ny,
                    const Args &args)
        :sites_(sites), nSite_(nSite),
         nx_(nx), ny_(ny),
         unitCell_(uc),
         dim_(boundD){
    isFerm_ = args.getBool("Fermionic",false);
    if(isFerm_) cout<<"Fermionic algorithm can only implement by IQIndex"<<endl;

    symInd_=Index("aux",1);
    auto symten = ITensor(symInd_);
    symten.set(symInd_(1),1.0);

    linksG_.clear();
    for (auto i:range(nSite_)) {
        linksG_.push_back(
                array<Index, 2>{{Index(nameint("UpIn", i), dim_, Link),
                                        Index(nameint("UpOut", i), dim_, Link)}});
        linksG_.push_back(
                array<Index, 2>{{Index(nameint("LeftIn", i), dim_, Link),
                                        Index(nameint("LeftOut", i), dim_, Link)}});
    }
    Ls_.clear();
    for (auto &I:linksG_) {
        Ls_.push_back(delta(I[0], I[1]));
    }
    G_.clear();
    for (auto i:range(nSite_)) {
        auto sitebond = siteBonds(i, nx_, ny_, unitCell_);
        auto Gtmp = randomTensor(dag(linksG_[sitebond[0]][0]),
                                 dag(linksG_[sitebond[1]][1]),
                                 dag(linksG_[sitebond[2]][1]),
                                 dag(linksG_[sitebond[3]][0]),
                                 sites_.si(i + 1));
        if(i==0) Gtmp *= symten;
        scaleTensor(Gtmp);
        G_.push_back(Gtmp);
    }
}


template<>
iPEPS<IQIndex>::iPEPS(const vector<int> & uc, const SiteSet &sites, const int nSite, int boundD, const int nx, const int ny,
                      const Args &args)
        :sites_(sites), nSite_(nSite),
         nx_(nx), ny_(ny),
         unitCell_(uc),
         dim_(boundD) {
    isFerm_ = args.getBool("Fermionic",false);

    auto symTen = symmtensor(QN());
    auto symInd_ = symTen.inds().front();

    linksG_.clear();
    for (auto i:range(nSite_)) {
        linksG_.push_back(
                array<IQIndex, 2>{{IQIndex(nameint("UpIn", i),
                                           Index("q-1", dim_ / 2, Link), QN(-1),
                                           Index("q0", dim_ / 2, Link), QN(0),
                                           Index("q1", dim_ / 2, Link), QN(1), In),
                                          IQIndex(nameint("UpOut", i),
                                                  Index("q-1", dim_ / 2, Link), QN(-1),
                                                  Index("q0", dim_ / 2, Link), QN(0),
                                                  Index("q1", dim_ / 2, Link), QN(1) )}});
        linksG_.push_back(
                array<IQIndex, 2>{{IQIndex(nameint("LeftIn", i),
                                           Index("q-1", dim_ / 2, Link), QN(-1),
                                           Index("q0", dim_ / 2, Link), QN(0),
                                           Index("q1", dim_ / 2, Link), QN(1), In),
                                          IQIndex(nameint("LeftOut", i),
                                                  Index("q-1", dim_ / 2, Link), QN(-1),
                                                  Index("q0", dim_ / 2, Link), QN(0),
                                                  Index("q1", dim_ / 2, Link), QN(1) )}});
    }
    Ls_.clear();
    for (auto &I:linksG_) {
        Ls_.push_back(delta(I[0], I[1]));
    }
    G_.clear();
    for (auto i:range(nSite_)) {
        auto sitebond = siteBonds(i, nx_, ny_, unitCell_);
        auto Gtmp = randomTensor(QN(),
                                 dag(linksG_[sitebond[0]][0]),
                                 dag(linksG_[sitebond[1]][1]),
                                 dag(linksG_[sitebond[2]][1]),
                                 dag(linksG_[sitebond[3]][0]),
                                 sites_.si(i + 1));
        if(i==0) Gtmp *= symTen;
        scaleTensor(G_[i]);
        G_.push_back(Gtmp);
    }
}

template<class IndexT>
void iPEPS<IndexT>::randomGs()
{
    for(auto &gs:G_)
        randomFill(gs);
}

template<class IndexT>
void iPEPS<IndexT>::AfromGandLs() {
    links_.clear();
    for (auto &I:linksG_)
        links_.push_back(I[1]);

    A_.clear();
    for (auto i:range(nSite_)) {
        A_.push_back(G_[i]);

        auto linkarray = siteBonds(i, nx_, ny_, unitCell_);
        for (auto j:range(4)) {
            auto D = Ls_.at(linkarray[j]);
            D.scaleTo(1.0);
            D.apply([](Real x) { return sqrt(x); });
            A_.at(i) *= D;
            switch (j) {
                case 1:
                case 2:
                    A_.at(i) *= delta(dag(linksG_.at(linkarray[j])[1]),
                                      dag(linksG_.at(linkarray[j])[0]));
                    break;
                default:
                    break;
            }
        }

        A_.at(i) /= maxAbs(A_.at(i));
        A_.at(i).scaleTo(1.0);
    }
}

// Apply op at site si
// 0<= si < nSite_
template<class IndexT>
void iPEPS<IndexT>::applySiteOp(const TensorT &op, const int si) {
    if( (si<0) || (si >= nSite_) ) cout<<"Can only apply op to 0<= site <= nSite_ "<<endl;
    G_.at(si) *= op;
    G_.at(si).noprime(Site);
    scaleTensor(G_[si]);
}

/*
 *                            prime   prime
 *                              |       |
 *                             L2      L4
 *                              |       |
 * tbond =           prime--L3--Ga--L1--Gb--L3--prime
 *                              |       |
 *                             L4      L2
 *                              |       |
 *                            prime   prime
 */
// Apply gate to bond and simple update. return the estimated energy.
// 0<= bond <= 2 * nSite_
template<class IndexT>
double iPEPS<IndexT>::applyGate(const TensorT &gate, const int bond) {
    if( (bond<0) || (bond >= 2*nSite_) )
        cout<<"Can only apply op to 0<= bond <= 2*nSite_ "<<endl;
    auto vert = (bond%2==0);
    auto indAB = calcABInd(bond, nx_, ny_,unitCell_);

    auto GA = G_.at(indAB[0]);
    auto GB = G_.at(indAB[1]);

    auto linkAarray = siteBonds(indAB[0], nx_, ny_, unitCell_);
    auto linkBarray = siteBonds(indAB[1], nx_, ny_, unitCell_);

    if(isFerm_)
    {
        if(vert)
        {
            if(indAB[0]==0) fSwapGate(GA,symInd_,findtype(GA,Site));
            fSwapGate(GA,linksG_[linkAarray[1]][1],findtype(GA,Site));
        }
        else 
        {
            if(indAB[1]==0) fSwapGate(GB,symInd_,findtype(GB,Site));
            fSwapGate(GA,linksG_[linkAarray[2]][1],findtype(GA,Site));
        }
    }

    for (auto i:range(4)) {
        if (linkAarray[i] == bond) continue;
        GA *= Ls_.at(linkAarray[i]);
    }
    for (auto i:range(4)) {
        if (linkBarray[i] == bond) continue;
        GB *= Ls_.at(linkBarray[i]);
    }

    auto tbond = GA * Ls_.at(bond) *GB;
    auto tEvBond = noprime(tbond * gate, Site);

    double e0 = (tEvBond * dag(tEvBond)).real()
                / (tbond * dag(tbond)).real();

    svd(tEvBond, GA, Ls_.at(bond), GB,
        {"Maxm", dim_, "LeftIndexName", nameint("Lin", bond),
         "RightIndexName", nameint("Lout", bond)});

    linksG_.at(bond) = array<IndexT, 2>{{commonIndex(Ls_.at(bond),GA),
                                                   commonIndex(Ls_.at(bond),GB)}};

    Ls_.at(bond) /= norm(Ls_.at(bond));
    Ls_.at(bond).scaleTo(1.0);

    for (auto i:range(4)) {
        if (linkAarray[i] == bond) continue;
        auto D = Ls_.at(linkAarray[i]);
        D.scaleTo(1.0);
        D.apply([](Real x) { return 1.0 / x; });
        GA *= dag(D);
    }
    for (auto i:range(4)) {
        if (linkBarray[i] == bond) continue;
        auto D = Ls_.at(linkBarray[i]);
        D.scaleTo(1.0);
        D.apply([](Real x) { return 1.0 / x; });
        GB *= dag(D);
    }
    scaleTensor(GA);
    scaleTensor(GB);

    if(isFerm_)
    {
        if(vert)
        {
            fSwapGate(GA,linksG_[linkAarray[1]][1],findtype(GA,Site));
            if(indAB[0]==0) fSwapGate(GA,symInd_,findtype(GA,Site));
        }
        else
        {

            fSwapGate(GA,linksG_[linkAarray[2]][1],findtype(GA,Site));
            if(indAB[1]==0) fSwapGate(GB,symInd_,findtype(GB,Site));
        }
    }

    G_.at(indAB[0]) = GA;
    G_.at(indAB[1]) = GB;

    return e0;
}

template class iPEPS<Index>;
template class iPEPS<IQIndex>;