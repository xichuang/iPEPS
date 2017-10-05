//
// Created by xich on 16-10-30.
//
#include "utilfunctions.h"
#include "itensor/itdata/qutil.h"
#include "itensor/decomp.h"
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <iomanip>
#include <random>

template<typename storagetype>
void printAll(storagetype const &d) {
    int i = 0;
    for (auto &I:d.store) {
        cout << i << "  " << setprecision(14) << I << endl;
        i++;
    }
}
template void printAll<DenseReal>(DenseReal const &d);
template void printAll<QDenseReal>(QDenseReal const &d);
template void printAll<DiagReal>(DiagReal const &d);
template void printAll<QDiagReal>(QDiagReal const &d);

Real cplxArg(const ITensor &T) {
    auto extractCplx = [](Dense<Cplx> const &d) {
        return d.store;
    };

    auto v = applyFunc(extractCplx, T.store());

    return arg(*(v.begin()));
}

template<typename TensorT>
Real maxAbs(const TensorT &T) {
    Real max_mag = 0.;
    auto maxComp = [&max_mag](Real r) {
        if (abs(r) > max_mag) max_mag = abs(r);
    };

    auto maxCompCplx = [&max_mag](Cplx r) {
        if (abs(r) > max_mag) max_mag = abs(r);
    };

    if (isComplex(T)) T.visit(maxCompCplx);
    else T.visit(maxComp);

    return max_mag;
}
template Real maxAbs<ITensor>(const ITensor &T);
template Real maxAbs<IQTensor>(const IQTensor &T);

template<typename TensorT>
void randomFill(TensorT& T)
{
    static default_random_engine e(time(NULL));
    static uniform_real_distribution<double> u(-1,1);
//    T.apply([&e,&u](Real x){return u(e);});
    T.apply([](Real x){return u(e);});
}
template void randomFill<ITensor>(ITensor &T);
template void randomFill<IQTensor>(IQTensor &T);

template<typename TensorT>
void scaleTensor(TensorT &T) {
    T/=maxAbs(T);
    T.scaleTo(1.0);
    if (isComplex(T)) {
        T.apply([](Cplx x) { return abs(x) < 1.0e-14 ? 0.0 : x; });
    }else{
        T.apply([](Real x) { return abs(x) < 1.0e-14 ? 0.0 : x; });
    }
}
template void scaleTensor<ITensor>(ITensor &T);
template void scaleTensor<IQTensor>(IQTensor &T);

Real entropy(const Spectrum &sp) {
    Real S = 0.0;
    for (int n = 1; n <= sp.numEigsKept(); ++n) {
        if (sp.eig(n) < 1.0e-10) continue;
        S += -sp.eig(n) * log(sp.eig(n));
    }
    return S;
}

Eigen::MatrixXd toEigen(ITensor T) {
    assert(T.r() < 3 && T.r() > 0);
    auto extractReal = [](Dense<Real> &d) {
        return d.store;
    };
    auto Tstore = applyFunc(extractReal, T.store());

    auto hasC = (T.r() == 1 ? false : true);

    Eigen::Map<Eigen::MatrixXd> mat(Tstore.data(), T.inds().front().m(),
                                    hasC ? T.inds().back().m() : 1);

    auto trans = (T.inds().front().primeLevel() != 0);
//    assert(!(trans&&hasC));
    Eigen::MatrixXd mat2 = mat;
//    if(trans) mat.transposeInPlace();
    if (trans) mat2 = mat.transpose();
    mat2 *= T.scale().real0();
//    return mat;
    return mat2;
}

ITensor toITensor(Eigen::MatrixXd mat, IndexSet inds) {
    int size;
    for (auto &I:inds)
        size += I.m();
    assert(size == mat.rows() * mat.cols());
    vector<Real> vec(mat.data(), mat.data() + mat.rows() * mat.cols());

    return ITensor(inds, Dense<Real>(move(vec)));
}


//solve(AX=B) for X
/*
void solveLinearEq(const ITensor &A, const ITensor &B, ITensor &X) {

    vector<Index> Alines;
    for (auto &I:A.inds())
        if (I.primeLevel() == 0) Alines.push_back(I);

    vector<Index> Bcols;
    for (auto &I:B.inds()) {
        if (I.primeLevel() == 0) Bcols.push_back(I);
    }
    auto Acomb = combiner(Alines);
    auto Bcomb = combiner(Bcols);

    auto tmpA = A * Acomb;
    tmpA *= prime(Acomb);
    auto tmpB = B * prime(Acomb);
    tmpB *= Bcomb;
    auto tmpX = X * Acomb;
    tmpX *= Bcomb;
    auto Rind = commonIndex(tmpB, prime(Acomb));
    auto Cind = commonIndex(tmpB, Bcomb);

    tmpX.prime(Cind);

    auto matA = toEigen(tmpA);
    auto matB = toEigen(swapPrime(tmpB, 0, 1));
//    Eigen::MatrixXd matX = matA.colPivHouseholderQr().solve(matB);
    auto matX = toEigen(tmpX);

    Eigen::MINRES<Eigen::MatrixXd> minres;
    minres.setTolerance(1.0e-10);
    minres.compute(matA);
    matX = minres.solveWithGuess(matB, matX);
    auto err = minres.error();
    if (abs(err) > 1.0e-8)
        cout << "warning: bad converge of minres with iterations: "
             << minres.iterations()
             << ", estimated error: "
             << err << endl;
    X = toITensor(matX, IndexSet(Rind, Cind));
    X *= prime(Acomb);
    X *= Bcomb;
    X.noprime();

//    Print(A*X-B);

    assert(norm(A * X - B) < 1.0e-12);
}
*/

template <typename TensorT>
void solveLinearEq(const TensorT &A, const TensorT &B, TensorT &X) {
    TensorT dR, eigR;
//    diagHermitian(A, dR, eigR,{"AbsoluteCutoff",true,"Cutoff",1.0e-10});

    diagHermitian(A, dR, eigR);
    eigR.scaleTo(1.0);
    /*
    Real elesum = 0.0;
    Real minele=1000.0;
    Real maxele=-1000.0;
    Real minAbs=1000;
    int numneg=0;
    int tnum=0;
    auto accum = [&elesum,&minele,&maxele,&minAbs,&numneg, &tnum](Real r)
    {
        if(r<0.0) numneg++;
        if(r<minele) minele=r;
        if(r>maxele) maxele=r;
        if(abs(r)<minAbs) minAbs=abs(r);
        elesum += r;
        tnum++;
    };
    eigR.visit(accum);
    cout<<" info of reduced tensor"<<endl;
    cout<<"min"<<" ~~ "<<"max"<<"  "<<"min of Abs"<<"  "<<"sum"<<" , "<<"number of negtive egs "<<" / "<<"total egs"<<endl;
    cout<< minele<<" ~~  "<<maxele<<"  "<<minAbs<<"  "<<elesum<<" ,  "<<numneg<<" / "<<tnum<<endl;


    diagHermitian(A, dR, eigR,{"Cutoff",1.0e-12});
     */
//    diagHermitian(A, dR, eigR);
//    PrintDat(eigR);
    eigR.apply([](Real x) { return 1.0 / x; });
//    eigR.apply([](Real x) { return 1.0/x;});

    auto invR = dag(dR) * eigR * prime(dR);
    X = noprime(swapPrime(invR, 0, 1) * B);
    assert(norm(A * X - B) < 1.0e-12);
}
template void solveLinearEq<ITensor>(const ITensor &A, const ITensor &B, ITensor &X);
template void solveLinearEq<IQTensor>(const IQTensor &A, const IQTensor &B, IQTensor &X);


//QR decomposition A=QR
void QRdecomp(const ITensor &A, ITensor &Q, ITensor &R,
              const bool buildQ, const Args& args) {
    auto iname = args.getString("IndexName","qr");
    auto itype = getIndexType(args,"IndexType",Link);
    /*
    ITensor D;
    svd(A, Q, D, R, {"Truncate",false});
    R *= D;
     */

    std::vector<Index> Qinds,
            Rinds;
//Divide up indices based on U
//If U is null, use V instead

    if (Q) {
        for (auto &I : A.inds()) {
            if (hasindex(Q, I))
                Qinds.push_back(I);
            else
                Rinds.push_back(I);
        }
    } else {
        for (auto &I : A.inds()) {
            if (hasindex(R, I))
                Rinds.push_back(I);
            else
                Qinds.push_back(I);
        }

    }

//    assert(Qinds.size() > Rinds.size());
    auto Qcomb = combiner(Qinds);
    auto Rcomb = combiner(Rinds);

    auto Atmp = A * Qcomb * Rcomb;
    Atmp.prime(commonIndex(Atmp, Rcomb));
    auto Amat = toEigen(Atmp);

    auto minInd=min(Amat.rows(),Amat.cols());

    auto qr = Eigen::HouseholderQR<Eigen::MatrixXd>(Amat);
    Eigen::MatrixXd Qmatfull,Qmat;
    if(buildQ) {
        Qmatfull = qr.householderQ();
        Qmat = Qmatfull.topLeftCorner(Amat.rows(), minInd);
    }
//    Eigen::MatrixXd Qmat = qr.householderQ();
    Eigen::MatrixXd Rmat = qr.matrixQR().topLeftCorner(minInd, Amat.cols()).triangularView<Eigen::Upper>();

    Index qrlink(iname, minInd, itype);
    if(buildQ) {
        Q = toITensor(Qmat, IndexSet(commonIndex(Atmp, Qcomb), qrlink));
        Q *= Qcomb;
    }
    R = toITensor(Rmat, IndexSet(qrlink, commonIndex(Atmp, prime(Rcomb))));
    R.noprime();
    R *= Rcomb;

    if(norm(A - Q * R)> 1.0e-12) cout<<"error in QR decomposition with relative error"
                                     <<norm(A-Q*R)<<endl;
}

//QR decomposition A=QR. Work only for IQTensor A with QN()
void QRdecomp(const IQTensor &A, IQTensor &Q, IQTensor &R,
              const bool buildQ, const Args& args) {
    auto iname = args.getString("IndexName","qr");
    auto itype = getIndexType(args,"IndexType",Link);

    auto sym=(div(A)==QN());

/*
    IQTensor D;
    svd(A,Q,D,R);
    R *= D;
    */
    std::vector<IQIndex> Qinds, Rinds;

    auto Atmp=A;

    auto auxTensor = symmtensor(-div(A));
    auto auxInd = auxTensor.inds().front();

    if(!sym) {
        cout << "warning! not a symmetric tensor!" << endl;
        Atmp *= auxTensor;
    }

//Divide up indices based on U
//If U is null, use V instead

    if (Q) {
        for (auto &I : A.inds()) {
            if (hasindex(Q, I))
                Qinds.push_back(I);
            else
                Rinds.push_back(I);
        }
    } else {
        for (auto &I : A.inds()) {
            if (hasindex(R, I))
                Rinds.push_back(I);
            else
                Qinds.push_back(I);
        }

    }

    if(!sym) {
        Rinds.push_back(auxInd);
    }

//    assert(Qinds.size() > Rinds.size());
    auto Qcomb = combiner(Qinds);
    auto Rcomb = combiner(Rinds);

//    auto Atmp = A * Qcomb * Rcomb;
    Atmp *= Qcomb;
    Atmp *= Rcomb;
//    Atmp = A * Qcomb * Rcomb;
    Atmp.scaleTo(1.0);
    auto uI=commonIndex(Atmp,Qcomb);
    auto vI=commonIndex(Atmp,Rcomb);

    auto blocks = doTask(GetBlocks<Real>{Atmp.inds(),uI,vI},Atmp.store());
    auto Nblock = blocks.size();
    if(Nblock == 0) throw ResultIsZero("IQTensor has no blocks");

    auto Qmats = vector<Mat<Real>>(Nblock);
    auto Rmats = vector<Mat<Real>>(Nblock);

    auto QRiq = IQIndex::storage{};
    QRiq.reserve(Nblock);
    auto trans = (uI==Atmp.inds().back());

    for(auto b : range(Nblock))
    {
        auto& B = blocks[b];
        auto& M = B.M;
        auto data=M.data();
        auto nr = trans?ncols(M):nrows(M);
        auto nc = trans?nrows(M):ncols(M);
        Eigen::Map<const Eigen::MatrixXd> mat(data, nr,
                                              nc);
        Eigen::MatrixXd mat2 = mat;
        if(trans) mat2.transposeInPlace();
        auto minInd=min(mat2.rows(),mat2.cols());
        QRiq.emplace_back(Index(iname, minInd, itype ),vI.qn(1+B.i2));

//        if(ncols(M)>nrows(M)) cout<<"error in QR decomposition, rows<cols"<<endl;
//        cout<<"mat dimensional ( "<< mat.rows()<<", "<<mat.cols()<<" )"<<endl;

        auto qr = Eigen::HouseholderQR<Eigen::MatrixXd>(mat2);
        Eigen::MatrixXd Qmatfull, Qmat;
        if(buildQ) {
            Qmatfull = qr.householderQ();
            Qmat = Qmatfull.topLeftCorner(mat2.rows(), minInd);
        }
        Eigen::MatrixXd Rmat = qr.matrixQR().topLeftCorner(minInd, mat2.cols()).triangularView<Eigen::Upper>();

        if(buildQ)
        {
            auto& QQ = Qmats.at(b);
            QQ = makeMatRef(Qmat.data(),Qmat.size(),Qmat.rows(),Qmat.cols());
        }
        auto& RR = Rmats.at(b);
        RR = makeMatRef(Rmat.data(),Rmat.size(),Rmat.rows(),Rmat.cols());
    }

    auto QRind = IQIndex(iname,move(QRiq),vI.dir());

    auto Qis = IQIndexSet(uI,QRind);
    auto Ris = IQIndexSet(dag(QRind),vI);

    auto Qstore = QDense<Real>(Qis,QN());
    auto Rstore = QDense<Real>(Ris,QN());

    long n = 0;
    for(auto b : range(Nblock))
    {
        auto& B = blocks[b];
        auto& UU = Qmats.at(b);
        auto& VV = Rmats.at(b);

        if(not B.M) continue;

        if(buildQ) {
            auto uind = stdx::make_array(B.i1, n);
            auto pU = getBlock(Qstore, Qis, uind);
            auto Uref = makeMatRef(pU, uI[B.i1].m(), QRind[n].m());
            Uref &= UU;
        }

        auto vind = stdx::make_array(n,B.i2);
        auto pV = getBlock(Rstore,Ris,vind);
        auto Vref = makeMatRef(pV,QRind[n].m(),vI[B.i2].m());
        Vref &= VV;

        ++n;
    }

    if(buildQ) {
        Q = IQTensor(Qis, move(Qstore));
        Q *= dag(Qcomb);
    }
    R = IQTensor(Ris,move(Rstore));
    R *= dag(Rcomb);

    if(!sym) {
        R *= dag(auxTensor);
    }

}

template <class IndexT>
void truncSVD(const ITensorT<IndexT> &A, ITensorT<IndexT> &U, ITensorT<IndexT> &D, ITensorT<IndexT> &V)
{
    std::vector<IndexT> Uinds, Vinds;
    for (auto &I : A.inds()) {
        if (hasindex(U, I))
            Uinds.push_back(I);
        else
            Vinds.push_back(I);
    }
    auto Ucomb = combiner(Uinds);
    auto Vcomb = combiner(Vinds);
    auto Atmp = A * Ucomb * Vcomb;

    auto tensorX=dag(U*Ucomb);
    auto mind=uniqueIndex(tensorX, Atmp, All);
    auto lind=commonIndex(Atmp, tensorX);
    auto rind=uniqueIndex(Atmp, tensorX, All);
    auto tensorY=tensorX*Atmp;
    ITensorT<IndexT> tensorYp,R=ITensorT<IndexT>(mind);
    QRdecomp(tensorY,tensorYp,R);
    auto tensorZ = Atmp*dag(tensorYp);
    U = ITensorT<IndexT>(lind);
    svd(tensorZ,U,D,V);
    V*=tensorYp;
    U *=dag(Ucomb);
    V *=dag(Vcomb);
}
template void truncSVD<Index>(const ITensor &A, ITensor &U, ITensor &D, ITensor &V);
template void truncSVD<IQIndex>(const IQTensor &A, IQTensor &U, IQTensor &D, IQTensor &V);

Spectrum nonSymSVD(const ITensor &A, ITensor &U,
               ITensor &D, ITensor &V, const Args& args)
{
    auto sp = svd(A,U,D,V,args);
    return sp;
}

IQTensor symmtensor(const QN qn)
{
    vector<IndexQN> iqtmp;
    iqtmp.push_back(IndexQN(Index("aux",1),qn));
    IQIndex auxInd = IQIndex("aux",move(iqtmp));
    IQTensor auxTensor = IQIndex(auxInd);
    auxTensor.set(auxInd(1),1.0);
    return auxTensor;
}

Spectrum nonSymSVD(const IQTensor &A, IQTensor &U,
               IQTensor &D, IQTensor &V, const Args& args)
{
    auto Ap=A;
    auto qn0=div(A);
    auto sym=(qn0==QN());
    auto auxTensor = symmtensor(-qn0);
    if(!sym) {
        Ap *= auxTensor;
    }
    auto sp = svd(Ap,U,D,V,args);
    if(!sym) {
        V *= dag(auxTensor);
    }

    return sp;

}

void matInverse(const ITensor &A, ITensor & invA)
{
    auto lind = A.inds().front();
    auto rind = A.inds().back();
    auto Ap = A*delta(rind,prime(lind));
    auto mat = toEigen(Ap);
    auto invmat=mat.inverse();

    auto inds = IndexSet(rind,lind);
    invA = toITensor(invmat,inds);

    assert(abs( norm(A*prime(invA,lind))-sqrt(lind.m()) )<1.0e-12);
    assert(abs( norm(A*prime(invA,rind))-sqrt(lind.m()) )<1.0e-12);
}
void matInverse(const IQTensor &A, IQTensor & invA)
{
    invA = A;
    invA.scaleTo(1.0);
    auto lind = A.inds().front();
    auto rind = A.inds().back();
    auto blocks = doTask(GetBlocks<Real>{invA.inds(),lind,rind},invA.store());
    auto Nblock = blocks.size();
    if(Nblock == 0) throw ResultIsZero("IQTensor has no blocks");

    auto invMats = vector<Mat<Real>>(Nblock);

    for(auto b : range(Nblock))
    {
        auto& B = blocks[b];
        auto& M = B.M;
        auto data=M.data();
        Eigen::Map<const Eigen::MatrixXd> mat(data, nrows(M),
                                              ncols(M));
        Eigen::MatrixXd invMat = mat.inverse();
        auto& imat = invMats.at(b);
        imat = makeMatRef(invMat.data(),invMat.size(),invMat.rows(),invMat.cols());
    }
    auto invIs = IQIndexSet(lind,rind);
    auto invStore = QDense<Real>(invIs,QN());

    long n = 0;
    for(auto b : range(Nblock))
    {
        auto& B = blocks[b];
        auto& invMat = invMats.at(b);
        if(not B.M) continue;

        auto uind = stdx::make_array(B.i1,n);
        auto pU = getBlock(invStore,invIs,uind);
        assert(pU.data() != nullptr);
        assert(uI[B.i1].m() == long(nrows(UU)));
        auto Uref = makeMatRef(pU,lind[B.i1].m(),rind[n].m());
        Uref &= invMat;
        ++n;
    }

    invA = IQTensor(invIs,move(invStore));
    invA *= qDelta(dag(lind),prime(dag(rind)));
    invA *= qDelta(dag(rind),dag(lind));
    invA.noprime();

    auto ntmp = abs( norm(A*prime(invA,lind))-sqrt(lind.m()) );
    if(ntmp>1.0e-12) cout<<"error in matrix inversion with N= "<< ntmp<<endl;
}


IQIndex cpInd(const IQIndex &cor, const Args& args) {
    auto iname = args.getString("IndexName",cor.name());
    auto itype = getIndexType(args,"IndexType",cor.type());
    vector<IndexQN> corInd;
    for (auto iq:cor) {
        auto ind = iq.index;
        auto qnum = iq.qn;
        auto cpind = cpInd(ind, args);
        corInd.push_back(IndexQN(cpind, qnum));
    }

    auto corcp = IQIndex(iname, move(corInd), cor.dir());

    return corcp;
}

Index cpInd(const Index &cor, const Args& args) {
    auto iname = args.getString("IndexName",cor.name());
    auto itype = getIndexType(args,"IndexType",cor.type());
    auto corcp = Index(iname, cor.m(), itype);
    return corcp;
}

void fSwapGate(ITensor &T, const Index &ind1, const Index &ind2)
{
    cout<<"fermionic algorithm can only apply to IQTensor"<<endl;
}
void fSwapGate(IQTensor &T, const IQIndex &ind1, const IQIndex &ind2) {
    assert(hasindex(T, ind1) && hasindex(T, ind2));

    vector<Index> oddInd1, oddInd2;
    for (auto i:range(ind1.nblock())) {
        if (paritySign(ind1.qn(i + 1)) == -1) {
            oddInd1.push_back(ind1.index(i + 1));
        }
    }

    for (auto i:range(ind2.nblock())) {
        if (paritySign(ind2.qn(i + 1)) == -1) {
            oddInd2.push_back(ind2.index(i + 1));
        }
    }

    auto oddT1 = ITensor(oddInd1), oddT2 = ITensor(oddInd2);

    auto applyFermGate = [&](QDense<Real> &d) {
        auto rofT = T.r();
        Labels block_ind(rofT);

        for (auto &io:d.offsets) {
            computeBlockInd(io.block, T.inds(), block_ind);
            vector<Index> blockInd;
            for (auto i:range(rofT))
                blockInd.push_back(T.inds().index(i + 1).index(block_ind.at(i) + 1));
            auto bIs = IndexSet(blockInd);

            if (commonIndex(ITensor(bIs), oddT1) && commonIndex(ITensor(bIs), oddT2)) {
                auto startInd = io.offset;
                auto size = 1;
                for (auto &I:blockInd)
                    size *= I.m();
                for (auto i:range(size))
                    d.store.at(startInd + i) *= -1.0;
            }
        }
    };

    applyFunc(applyFermGate, T.store());
}

ITensor qDelta(const Index &ind1, const Index &ind2) {
    return delta(ind1, ind2);
}

IQTensor qDelta(const IQIndex &ind1, const IQIndex &ind2, const QN q) {
    auto is = IQIndexSet(ind1, ind2);
    auto dat = QDenseReal{is, q};
    auto T = IQTensor(move(is), move(dat));

    auto fillDelta = [&](QDense<Real> &d) {
        auto rofT = T.r();
        Labels block_ind(rofT);

        for (auto &io:d.offsets) {
            computeBlockInd(io.block, T.inds(), block_ind);
            vector<Index> blockInd;
            for (auto i:range(rofT))
                blockInd.push_back(T.inds().index(i + 1).index(block_ind.at(i) + 1));

            if ((blockInd.at(0).m()) != (blockInd.at(1).m()))
                cout << "warning: not a square delta function." << endl;
            auto numele = min(blockInd.at(0).m(), blockInd.at(1).m());
            for (auto i:range(numele)) {
                auto indstart = io.offset;
                d.store.at(indstart + i * blockInd.at(0).m() + i) = 1.0;
            }
        }
    };

    applyFunc(fillDelta, T.store());

    return T;
}

bool isEqual(const Index& ind1, const Index& ind2)
{
    return ind1.m()==ind2.m();
}

bool isEqual(const IQIndex& ind1, const IQIndex& ind2)
{
    if(ind1.m() != ind2.m()) return false;
    if(ind1.nblock() != ind2.nblock()) return false;

    vector<IndexQN> iqInd1, iqInd2;
    if(ind1.dir()==In) {
        for (auto iq:ind1)
            iqInd1.push_back(iq);
    } else{
        for( auto iq:ind1)
            iqInd1.push_back(IndexQN(iq.index,-iq.qn));
    }
    if(ind2.dir()==Out) {
        for (auto iq:ind2)
            iqInd2.push_back(iq);
    } else{
        for( auto iq:ind2)
            iqInd2.push_back(IndexQN(iq.index,-iq.qn));
    }

//    bool operator<[](IndexQN const & a, IndexQN const & b){return a.qn <b.qn;}
//    bool operator==[](IndexQN const & a, IndexQN const & b){return (a.qn == b.qn)&& isEqual(a.index,b.index);}
    sort(iqInd1.begin(), iqInd1.end(), [](IndexQN const & a, IndexQN const & b){return a.qn <b.qn;});
    sort(iqInd2.begin(), iqInd2.end(), [](IndexQN const & a, IndexQN const & b){return a.qn <b.qn;});

    for(auto a=iqInd1.begin(),b=iqInd2.begin();
        a!=iqInd1.end(),b!=iqInd2.end();
        ++a, ++b)
    {
        if(!((*a).qn == (*b).qn && isEqual((*a).index,(*b).index))) return false;
    }
    return true;
}

// return nearest sites A and B of bond.
// 0<= bond <= 2 * nSite_
// return 0<= si(A), si(B) < nSite_
array<int,2> calcABInd(const int bond, const int nx, const int ny, const vector<int>& uc)
{
    int lineA,lineB,colA,colB;
    lineA = (bond/2)/nx;
    colB = (bond/2)%nx;
    auto vert = (bond%2==0);
    if(vert)
    {
        colA=colB;
        lineB = lineA-1;

    }
    else{
        colA=colB-1;
        lineB=lineA;
    }

    auto indA= uc[indLC(lineA,colA,nx,ny)];
    auto indB= uc[indLC(lineB,colB,nx,ny)];

    return array<int,2>{{indA,indB}};
};

// return nearest bonds of site si.
// 0<= si < nSite_
// return 0<= (upbond, leftbond, downbond, rightbond)< 2*nSite_
array<int,4> siteBonds(const int si, const int nx, const int ny, const vector<int>& uc)
{
    int line = si/nx;
    int col = si%nx;
    auto upbond = 2*si;
    auto leftbond = 2*si+1;
    auto downbond = 2*(uc[indLC(line+1,col,nx,ny)]);
    auto rightbond = 2*(uc[indLC(line,col+1,nx,ny)])+1;

    array<int,4> linkarray={upbond,leftbond,downbond,rightbond};

    return linkarray;
}

// return site label at (line, col)
// 0<= line< ny;
// 0<= col < nyx;
// return 0<= si < nx*ny
int indLC(const int line,const int col,const int nx,const int ny)
{
    auto linep=line;
    auto colp=col;
    while (linep<0) linep += ny;
    while (colp<0) colp += nx;

    return (linep%ny)*nx + colp%nx;
}

template <class IndexT>
IndexT commonUniq(const ITensorT<IndexT> &T1,const ITensorT<IndexT> &T2,const ITensorT<IndexT> &T3)
{
    IndexT ind;
    for(auto &I:T1.inds())
    {
        if(hasindex(T2,I)&&(!hasindex(T3,I))){
            ind = I;
            break;
        }
    }
    return ind;
}

template Index commonUniq<Index>(const ITensor &T1,const ITensor &T2,const ITensor &T3);
template IQIndex commonUniq<IQIndex>(const IQTensor &T1,const IQTensor &T2,const IQTensor &T3);