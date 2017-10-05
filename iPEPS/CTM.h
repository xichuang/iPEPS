//
// Created by xich on 16-11-17.
//

#ifndef IPEPS_CTM_H
#define IPEPS_CTM_H

#include <vector>
#include <assert.h>

#include "itensor/all.h"
#include "utils/utilfunctions.h"
#include <utils/pstrickCTM.h>

using namespace std;
using namespace itensor;

template <class IndexT>
class CTM;

using CTM_ITensor  = CTM<Index>;
using CTM_IQTensor  = CTM<IQIndex>;

template <class IndexT>
class CTM {
    using TensorT=ITensorT<IndexT>;
public:
    CTM(const vector<int> &uc, const vector<TensorT>& As,
        const vector<IndexT>& links, const IndexT & symInd,
        const int nsite=2, const int nx=2, const int ny=2, const int chi=20,
        const Args& args=Global::args());
    void EnvFromAs();
    void randEnv();

    vector<TensorT> tensorAs() const{return As_;}
    IndexT symInd() const{return symInd_;}
    vector<TensorT> tensorAAs() const{return AAs_;}
    vector<IndexT> linkAs() const{return links_;}
    vector<TensorT> combAAs() const{return combAB_;}
    void drawEnvBulk(string name="ctm_", bool print=false);

    // return CTM tensors with the left up bulk tensor being site si.
    //  0<= si < nx*ny
    vector<TensorT> returnEnv(const int si)
    {
        if( si<0 || si>= (nx_*ny_))
            cout<< "can only return env with 0<=site< nx*ny"<<endl;
        if(si!=constructed_) constructBulkEnv(si);
        vector<TensorT> env;
        for(auto i:range(12)) {
            if (i % 3 == 0) env.push_back(Cs_[envT_[i][0]][envT_[i][1]]);
            else env.push_back(Ts_[envT_[i][0]][envT_[i][1]]);
        }
        return env;
    }

    // return bulk tensors with the left up bulk tensor being site si.
    //  0<= si < sx*xy
    vector<TensorT> returnBulk(const int si)
    {
        if( si<0 || si>= (nx_*ny_))
            cout<< "can only return env with 0<=site< nx*ny"<<endl;

        if(si!=constructed_) constructBulkEnv(si);
        vector<TensorT> bulk;
        for(auto i:range(4))
            bulk.push_back(AAs_[bulkT_[i]]);
        return bulk;
    }

    Real CTMRG(const int maxRGStep=20, const Real error=1.0e-10);

    Real exp_site(const TensorT &op, const int si);
    Real exp_bond(const TensorT &op, const int bond);

    Real reduceUpdate(const TensorT &gate, const int bond,
                      const int maxloop=20, const Real error=1.0e-12);
    void applySiteOp(const TensorT &op, const int si);

    Real fastUpdate(const TensorT &gate, const int bond,
                      const int maxloop=20, const Real error=1.0e-12);
    
    void checkNorm();

    Real stateNorm()
    {
        if(updatedEnv_<0) bondEnv(0);
        return norm_.real();
    }

    void resetBondD(int bondD){bondD_ = bondD;}
    void resetEnvD(int chi){chi_ = chi;}

private:

    void combineBond(TensorT & T, const int si, const int bond=-1);
    void extractBond(TensorT & T, const int si, const int bond=-1);
    void AAfromA(TensorT &AA, const TensorT &A, const int site );
    void AAfromAs();
    void scaleAs();
    void scaleEnv();

    void constructBulkEnv(const int si);
    void resetEnvLink(const int bond);

    array<TensorT,2 > projOP(const vector<TensorT>& Qs, const array<IndexT,2> Qind, const array<IndexT,2> &Rind );
    vector<TensorT> leftCTMRGStep(const int col, const bool his=false);
    vector<TensorT> rightCTMRGStep(const int col, const bool his=false);
    vector<TensorT> upCTMRGStep(const int line, const bool his=false);
    vector<TensorT> downCTMRGStep(const int line, const bool his=false);
    void CTMRGstep();
    vector<Real> cornerES() const;

    //factor A_, B_ to A_=AX*aR, B_=BY*bL;
    void reduceAB(TensorT& AX, TensorT& BY, TensorT& aR,TensorT& bL,
                  TensorT& AAX,TensorT& BBY, const int bond);
    void bondEnv(const int bond);
    void fixGauge(const int bond, TensorT &tildeAR,TensorT &tildeBL,
                  TensorT &tildeN,TensorT &invL,TensorT &invR);
    Real updateAB(const TensorT &gate, const int bond, IndexT &cind,
                  const int maxloop=20, const Real error=1.0e-12);

    const int nx_;
    const int ny_;
    const int nSite_;
    const vector<int> unitCell_;
    int chi_,bondD_;
    bool isFerm_;
    vector<IndexT> links_;
    IndexT symInd_;
    vector<TensorT> As_,AAs_;
    TensorT A_, B_, AA_, BB_;

    vector<TensorT> combAB_;

    vector<array<int,2>> envT_;
    vector<int> bulkT_;

    vector< array<TensorT,4> > Cs_,Ts_;

    int constructed_, updatedEnv_;

    TensorT AX_,BY_,aR_,bL_, bondenv_, norm_;

    array<int,2> abInd_;

};


#endif //IPEPS_CTM_QN_H
