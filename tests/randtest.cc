//
// Created by xich on 17-3-3.
//
#include "utils/utilfunctions.h"
#include <sys/time.h>

using namespace itensor;
using namespace std;
int main()
{
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
    PrintDat(A);
    randomFill(A);
    PrintDat(A);
    randomFill(A);
    PrintDat(A);
    randomFill(A);
    PrintDat(A);

    return 0;
}