//
// Created by xich on 17-1-14.
//

#ifndef IPEPS_ARNOLDI_H
#define IPEPS_ARNOLDI_H

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>
#include "itensor/all.h"
#include "utilfunctions.h"
#include <vector>

using namespace std;
using namespace itensor;

template<typename TensorT>
void arnoldi(TensorT &op, TensorT & vec, const Args& args)
{
    kmax=20;
    Eigen::MatrixXcd hmat(kmax, kmax);
    vector< TensorT > qs;
    vec/=norm(vec);
    qs.push_back(vec);

    for(auto k=1;k<kmax;++k)
    {
        qs.push_back(noprime(op*qs[k-1]));
        for(auto j=0;j<k;++j)
        {
            hmat(j,k-1) = (dag(qs[j])*qs[k]).real();
            qs[k] = qs[k] - hmat(j,k-1)*qs[j];
        }
        hmat(k,k-1) = norm(qs[k]);
        qs[k]/=hmat(k,k-1);
    }



}
#endif //IPEPS_ARNOLDI_H
