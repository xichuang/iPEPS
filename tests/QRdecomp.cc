//
// Created by xich on 17-2-3.
//
#include "utils/utilfunctions.h"
#include <sys/time.h>

using namespace itensor;
using namespace std;
int main()
{
    struct timeval start1,start2,start3,start4;
    struct timeval end1,end2,end3,end4;

    vector<IndexQN> iq1,iq2;
    iq1.push_back(IndexQN(Index("l-1",10),QN(-1)));
    iq1.push_back(IndexQN(Index("l-0",10),QN(0)));
    iq1.push_back(IndexQN(Index("l+1",10),QN(1)));

    iq2.push_back(IndexQN(Index("r-1",10),QN(-1)));
    iq2.push_back(IndexQN(Index("r-0",10),QN(0)));
    iq2.push_back(IndexQN(Index("r+1",10),QN(1)));

    IQIndex lind=IQIndex("lind",move(iq1));
    IQIndex rind=IQIndex("rind",move(iq2));

    IQTensor A=randomTensor(QN(),rind,lind);
    IQTensor B=randomTensor(QN(),lind,rind);

    IQTensor C=randomTensor(QN(1),lind,rind);

    IQTensor Q= IQTensor(lind), R;
    QRdecomp(A,Q,R);
    Print(norm(A-Q*R));

    IQTensor Qc= IQTensor(lind), Rc;
    QRdecomp(C,Qc,Rc);
    Print(norm(C-Qc*Rc));

    IQTensor D;
    gettimeofday(&start1,NULL);
    svd(A,Q,D,R);
    gettimeofday(&end1,NULL);
    cout<<"svd time:"
        <<1000000 * (end1.tv_sec - start1.tv_sec) + (end1.tv_usec - start1.tv_usec)<<endl;

    gettimeofday(&start2,NULL);
    QRdecomp(B,Q,R);
    gettimeofday(&end2,NULL);
    cout<<"Full QR time:"
        <<1000000 * (end2.tv_sec - start2.tv_sec) + (end2.tv_usec - start2.tv_usec)<<endl;
    Print(norm(B-Q*R));

    gettimeofday(&start3,NULL);
    QRdecomp(B,Q,R, false);
    gettimeofday(&end3,NULL);
    cout<<"Full QR time:"
        <<1000000 * (end3.tv_sec - start3.tv_sec) + (end3.tv_usec - start3.tv_usec)<<endl;
    Print(R);

    return 0;
}