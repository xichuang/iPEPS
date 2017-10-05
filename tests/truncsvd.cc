//
// Created by xich on 17-2-3.
//
#include "utils/utilfunctions.h"
#include <ctime>

using namespace itensor;
using namespace std;
int main()
{
    clock_t tbegin,tend;

    vector<IndexQN> iq1,iq2;
    iq1.push_back(IndexQN(Index("l-1",400),QN(-1)));
    iq1.push_back(IndexQN(Index("l-0",600),QN(0)));
    iq1.push_back(IndexQN(Index("l+1",800),QN(1)));

    iq2.push_back(IndexQN(Index("r-1",400),QN(-1)));
    iq2.push_back(IndexQN(Index("r-0",600),QN(0)));
    iq2.push_back(IndexQN(Index("r+1",800),QN(1)));

    IQIndex lind=IQIndex("lind",move(iq1));
    IQIndex rind=IQIndex("rind",move(iq2));

    IQTensor A=randomTensor(QN(),lind,rind);
    auto maxm=100;

    IQTensor U= IQTensor(lind), D, V;
    tbegin=std::clock();
    svd(A,U, D, V,{"Maxm",maxm});
    tend=std::clock();
    cout<<"ITensor SVD time:    "
        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
    Print(norm(A-U*D*V));

    PrintDat(D);

    tbegin=std::clock();

    auto mind=IQIndex("mind",Index("l-1",maxm),QN(-1));
    auto U0=IQTensor(lind);
    denmatDecomp(A,U0,V,Fromleft,{"Maxm",maxm});
    for(auto i:range(1))
        truncSVD(A,U0,D,V);
    tend=std::clock();
    cout<<"truncate SVD time:    "
        <<(tend - tbegin)*1.0/CLOCKS_PER_SEC<<endl;
    Print(norm(A-U0*D*V));
    PrintDat(D);

    return 0;
}