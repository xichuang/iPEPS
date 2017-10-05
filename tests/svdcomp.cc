//
// Created by xich on 17-2-3.
//

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#include "utils/utilfunctions.h"
#include "utils/RedSVD.h"
#include <ctime>

using namespace std;
using namespace itensor;
int main()
{
    clock_t tbegin,tend;
    int dim = 6;
    int chi=10*dim;
    int m=dim*dim*chi;
    int n = m;
    cout<<"matrix dimentional: ( "<<m<<" , "<<n<<" )"<<endl;
    int maxm=chi;
    auto lind = Index("lind",m);
    auto rind = Index("rind",n);

    ITensor A = randomTensor(lind,rind);
    ITensor S = ITensor(lind), D=ITensor(rind);
    ITensor V;
    tbegin=std::clock();
    svd(A,S,V,D);
    tend=std::clock();
    cout<<"ITensor Full SVD time:    "
        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
//    PrintDat(V);

    S = ITensor(lind);
    D = ITensor(rind);
    tbegin=std::clock();
    denmatDecomp(A,S,D, Fromleft,{"Maxm",maxm});
    S = ITensor(lind);
    D = ITensor(rind);
    denmatDecomp(A,S,D, Fromright,{"Maxm",maxm});
    tend=std::clock();
    cout<<"ITensor denmatDecomp time:"
        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;

//    PrintDat(V);
    ITensor V0;
    ITensor S0 = ITensor(lind);
    ITensor D0 = ITensor(rind);
    tbegin=std::clock();
    svd(A,S0,V0,D0,{"Maxm",maxm});
    tend=std::clock();
    cout<<"ITensor trancate SVD time:"
            <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;

    PrintDat(V0);
    Print(norm(A-S0*V0*D0));

    auto mat = toEigen(prime(A,rind));
    tbegin=std::clock();
    Eigen::BDCSVD<Eigen::MatrixXd> bdcsvd(mat,Eigen::ComputeThinU | Eigen::ComputeThinV);
//    auto matU=bdcsvd.matrixU();
//    auto matV=bdcsvd.matrixV();
    tend=std::clock();
    cout<<"BDCSVD SVD time:          "
            <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
//    Print(bdcsvd.singularValues());

    tbegin=std::clock();;
    Eigen::JacobiSVD<Eigen::MatrixXd>
            jacsvd(mat,Eigen::ComputeThinU | Eigen::ComputeThinV);
//    auto matJU=jacsvd.matrixU();
//    auto matJV=jacsvd.matrixV();
    tend=std::clock();
    cout<<"JacobiSVD SVD time:       "
            <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
    for(auto i:range(4))
        cout<<(jacsvd.singularValues()(i))<<endl;

    tbegin = std::clock();;
    RedSVD::RedSVD<Eigen::MatrixXd> rsvd(mat,4);
//    auto matJU=jacsvd.matrixU();
//    auto matJV=jacsvd.matrixV();
    tend=std::clock();
    cout<<"Random SVD time:       "
        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
    Print(rsvd.singularValues());

    tbegin=std::clock();
    ITensor Vp, Sp, Dp;
    Sp=ITensor(lind);
    denmatDecomp(A,Sp,Dp, Fromleft,{"Maxm",maxm});
    truncSVD(A, Sp, Vp, Dp);
    tend=std::clock();
    cout<<"truncate SVD time:       "
        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;

    PrintDat(Vp);
    Print(norm(A-Sp*Vp*Dp));


    return 0;
}
