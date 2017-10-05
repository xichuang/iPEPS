//
// Created by xich on 17-1-28.
//

#include "utils/utilfunctions.h"

using namespace itensor;
using namespace std;
int main()
{
    vector<IndexQN> iq1,iq2, iq3;
    iq1.push_back(IndexQN(Index("a-1",2),QN(-1)));
    iq1.push_back(IndexQN(Index("a-0",3),QN(0)));
    iq1.push_back(IndexQN(Index("a+1",4),QN(1)));

    iq2.push_back(IndexQN(Index("b+1",4),QN(1)));
    iq2.push_back(IndexQN(Index("b-0",3),QN(0)));
    iq2.push_back(IndexQN(Index("b-1",2),QN(-1)));

    iq3.push_back(IndexQN(Index("i-1",4),QN(-1)));
    iq3.push_back(IndexQN(Index("i-0",3),QN(0)));
    iq3.push_back(IndexQN(Index("i+1",2),QN(1)));

    auto ind1 = IQIndex("ind1", move(iq1));
    auto ind2 = IQIndex("ind1", move(iq2));
    auto ind3 = IQIndex("ind1", move(iq3));

    cout<< isEqual(ind1,dag(ind2))<<endl;
    cout<< isEqual(ind1,ind2)<<endl;
    cout<< isEqual(ind1,ind3)<<endl;

    Print(qDelta(ind1,ind2));

    Print(qDelta(ind1,dag(ind2)));

    Print(qDelta(ind1,ind3));

    return 0;
}