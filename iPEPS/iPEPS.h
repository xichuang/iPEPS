//
// Created by xich on 16-10-19.
// iPEPS with simple update algorithm.
//

#ifndef IPEPS_H
#define IPEPS_H

#include <vector>
#include <array>
#include <assert.h>

#include "itensor/all.h"
#include "utils/utilfunctions.h"

using namespace std;
using namespace itensor;

template<class IndexT>
class iPEPS;

using iPEPS_ITensor  = iPEPS<Index>;
using iPEPS_IQTensor  = iPEPS<IQIndex>;

template<class IndexT>
class iPEPS {
    using TensorT=ITensorT<IndexT>;
public:
    iPEPS(const vector<int> &uc, const SiteSet& sites, const int nSite=2,
          int boundD=2, const int nx=2, const int ny=2,
          const Args& args=Global::args());
    iPEPS(const vector<int> &uc, const SiteSet& sites,
          vector< array<IndexT,2> > &linksG, vector<TensorT> &Ls,
          vector<TensorT> &G, IndexT &symInd,
          const int nSite=2, int boundD=2, const int nx=2, const int ny=2,
          const Args& args=Global::args());

    void randomGs();
    void AfromGandLs();
    void applySiteOp(const TensorT & op, const int si);
    double applyGate(const TensorT & gate, const int bond);

    vector<TensorT> tensorAs() const{ return A_;}
    vector<IndexT> linkAs() const{return links_;}
    IndexT symInd() const{return symInd_;}

    int dim() const{ return dim_;}
    void resetBondD(int dimp){ dim_ = dimp;}

private:

    const SiteSet& sites_;
    const int nSite_,nx_,ny_;
    int dim_;
    const vector<int> unitCell_;
    bool isFerm_;

    vector<IndexT> links_;
    vector< array<IndexT,2> > linksG_;
    vector<TensorT> Ls_;
    vector<TensorT> G_,A_;
    IndexT symInd_;
};

#endif //iPEPS_SU_H
