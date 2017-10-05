//
// Created by xich on 16-10-30.
//

#ifndef IPEPS_UTILFUNCTIONS_H
#define IPEPS_UTILFUNCTIONS_H

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "itensor/all.h"
using namespace std;
using namespace itensor;

Real cplxArg(const ITensor& T);

template <typename TensorT>
Real maxAbs(const TensorT&);
template <typename TensorT>
void scaleTensor(TensorT&);

template <typename TensorT>
void randomFill(TensorT&);

Real entropy(const Spectrum&);

Eigen::MatrixXd toEigen(ITensor);
ITensor toITensor(Eigen::MatrixXd, IndexSet inds);

//solve(AX=B) for X
template <typename TensorT>
void solveLinearEq(const TensorT&, const TensorT &, TensorT &);

//QR decomposition
void orderIQind(IQTensor &A, const IQIndex& uI, const IQIndex& vI);
void QRdecomp(const ITensor &A, ITensor &Q, ITensor &R,
              const bool buildQ= true, const Args& args=Global::args());
void QRdecomp(const IQTensor &A, IQTensor &Q, IQTensor &R,
              const bool buildQ= true, const Args& args=Global::args());

//truncate SVD
template <class IndexT>
void truncSVD(const ITensorT<IndexT> &A, ITensorT<IndexT> &U,
              ITensorT<IndexT> &D, ITensorT<IndexT> &V);

// SVD of non-symmetric tensor
IQTensor symmtensor(const QN qn);
Spectrum nonSymSVD(const ITensor &A, ITensor &U,
               ITensor &D, ITensor &V, const Args& args=Global::args());
Spectrum nonSymSVD(const IQTensor &A, IQTensor &U,
               IQTensor &D, IQTensor &V, const Args& args=Global::args());

// Inversion of trianglular matrix A
void matInverse(const ITensor &A, ITensor & invA);
void matInverse(const IQTensor &A, IQTensor & invA);

//make a copy of index with the same dimensional and QN
IQIndex cpInd(const IQIndex& cor, const Args& args=Global::args());
Index cpInd(const Index& cor, const Args& args=Global::args());

void fSwapGate(ITensor &T, const Index &ind1, const Index &ind2);
void fSwapGate(IQTensor & T, const IQIndex & ind1, const IQIndex & ind2);

ITensor qDelta(const Index& ind1, const Index& ind2);
IQTensor qDelta(const IQIndex& ind1, const IQIndex& ind2, const QN q=QN() );

bool isEqual(const Index& ind1, const Index& ind2);
bool isEqual(const IQIndex& ind1, const IQIndex& ind2);

template<typename storagetype>
void printAll( storagetype const &d);

array<int,2> calcABInd(const int bond, const int nx, const int ny, const vector<int>& uc);
array<int,4> siteBonds(const int si, const int nx, const int ny, const vector<int>& uc);
int indLC(const int line,const int col,const int nx,const int ny);

template <class IndexT>
IndexT commonUniq(const ITensorT<IndexT> &T1,const ITensorT<IndexT> &T2,const ITensorT<IndexT> &T3);
#endif //IPEPS_UTILFUNCTIONS_H
