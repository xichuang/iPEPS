//
// Created by xich on 16-12-6.
//

#ifndef IPEPS_PSTRICKCTM_H
#define IPEPS_PSTRICKCTM_H

#include "itensor/all.h"
#include "utilfunctions.h"
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
using namespace itensor;

//bulkLink:
// b--1--b
// |     |
// 0     2
// |     |
// b--3--b
//
template<typename IndexT>
void pstrickCTM(const vector<ITensorT<IndexT>> &env, const vector<ITensorT<IndexT>> &bulk,
                const vector<IndexT> &bulkLink,
                const string fname="ctm.tex") {
    fstream psout;
    psout.open(fname,ios::out);
    assert(psout.is_open());

    psout << "\\psscalebox{1.0 1.0} % Change this value to rescale the drawing." << endl;
    psout << "{" << endl;
    psout << "\\begin{pspicture}(0,-5.5360937)(10.92,5.5360937)" << endl;
    psout << "\\definecolor{colour5}{rgb}{0.972549,0.039215688,0.02745098}" << endl;
    psout << "\\definecolor{colour6}{rgb}{0.98039216,0.019607844,0.019607844}" << endl;
    psout << "\\definecolor{colour7}{rgb}{0.08235294,0.07058824,0.9882353}" << endl;
    psout << "\\definecolor{colour8}{rgb}{0.07450981,0.05882353,0.98039216}" << endl;
    psout << "\\definecolor{colour9}{rgb}{0.0627451,0.019607844,0.019607844}" << endl;
    psout << "\\definecolor{colour10}{rgb}{0.94509804,0.67058825,0.015686275}" << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](0.4,4.95){0.4} "
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](3.6,4.95){0.4} "
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](6.8,4.95){0.4} "
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](10.0,4.95){0.4}"
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](0.4,1.75){0.4} "
            << endl;
    psout
            << "\\pscircle[linecolor=colour7, linewidth=0.08, fillstyle=solid,fillcolor=colour8, dimen=outer](3.6,1.75){0.4} "
            << endl;
    psout
            << "\\pscircle[linecolor=colour7, linewidth=0.08, fillstyle=solid,fillcolor=colour8, dimen=outer](6.8,1.75){0.4} "
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](10.0,1.75){0.4}"
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](0.4,-1.45){0.4}"
            << endl;
    psout
            << "\\pscircle[linecolor=colour7, linewidth=0.08, fillstyle=solid,fillcolor=colour8, dimen=outer](3.6,-1.45){0.4}"
            << endl;
    psout
            << "\\pscircle[linecolor=colour7, linewidth=0.08, fillstyle=solid,fillcolor=colour8, dimen=outer](6.8,-1.45){0.4}"
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](10.0,-1.45){0.4}"
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](0.4,-4.65){0.4}"
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](3.6,-4.65){0.4}"
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](6.8,-4.65){0.4}"
            << endl;
    psout
            << "\\pscircle[linecolor=colour5, linewidth=0.08, fillstyle=solid,fillcolor=colour6, dimen=outer](10.0,-4.65){0.4}"
            << endl;
    psout << "\\psframe[linecolor=colour9, linewidth=0.08, dimen=outer](10.0,4.95)(0.4,-4.65)" << endl;
    psout << "\\psline[linecolor=colour9, linewidth=0.08](0.4,1.75)(10.0,1.75)(10.0,1.75)" << endl;
    psout << "\\psline[linecolor=colour9, linewidth=0.08](0.4,-1.45)(10.0,-1.45)" << endl;
    psout << "\\psline[linecolor=colour9, linewidth=0.08](3.6,4.95)(3.6,-4.65)" << endl;
    psout << "\\psline[linecolor=colour9, linewidth=0.08](6.8,4.95)(6.8,-4.65)" << endl;

    vector<IndexT> horiInds;
    vector<IndexT> vertInds;

    horiInds.push_back(commonIndex(env[1], env[0]));
    horiInds.push_back(commonUniq(env[2], env[1], env[0]));
    horiInds.push_back(commonIndex(env[3], env[2]));

    horiInds.push_back(commonIndex(bulk[0], env[11]));
    horiInds.push_back(bulkLink[1]);
    horiInds.push_back(commonIndex(env[4], bulk[1]));

    horiInds.push_back(commonIndex(bulk[2], env[10]));
    horiInds.push_back(bulkLink[3]);
    horiInds.push_back(commonIndex(env[5], bulk[3]));

    horiInds.push_back(commonIndex(env[8], env[9]));
    horiInds.push_back(commonUniq(env[7], env[8], env[9]));
    horiInds.push_back(commonIndex(env[6], env[7]));

    vertInds.push_back(commonIndex(env[11], env[0]));
    vertInds.push_back(commonUniq(env[10], env[11], env[0]));
    vertInds.push_back(commonIndex(env[9], env[10]));

    vertInds.push_back(commonIndex(bulk[0], env[1]));
    vertInds.push_back(bulkLink[0]);
    vertInds.push_back(commonIndex(env[8], bulk[2]));

    vertInds.push_back(commonIndex(bulk[1], env[2]));
    vertInds.push_back(bulkLink[2]);
    vertInds.push_back(commonIndex(env[7], bulk[3]));

    vertInds.push_back(commonIndex(env[4], env[3]));
    vertInds.push_back(commonUniq(env[5], env[4], env[3]));
    vertInds.push_back(commonIndex(env[6], env[5]));

    auto dxy = 3.2;
    auto arrowLen = 0.8;

    auto x0 = 1.6;
    auto y0 = 5.35;
    auto arrowX0 = 1.6;
    auto arrowY0 = 4.95;
    for (auto i:range(12)) {
        auto line = i / 3;
        auto col = i % 3;

        auto xaxis = x0 + col * dxy;
        auto yaxis = y0 - line * dxy;
        auto arrowX = arrowX0 + col * dxy;
        auto arrowY = arrowY0 - line * dxy;

        auto arrowdir = (horiInds.at(i).dir() == In) ? "->" : "<-";

        psout << "\\psline[linecolor=colour10, linewidth=0.08, arrowsize=0.053cm 4.0,arrowlength=2.0,arrowinset=0.0]{"
             << arrowdir << "}(" << arrowX << "," << arrowY << ")("
             << arrowX + arrowLen << "," << arrowY << ")" << endl;

        psout << "\\rput[bl](" << xaxis
             << "," << yaxis << "){"
             << horiInds.at(i).id() % 1000 << "}" << endl;
    }
    x0 = 0.8;
    y0 = 3.35;
    arrowX0 = 0.4;
    arrowY0 = 3.75;
    for (auto i:range(12)) {
        auto line = i%3 ;
        auto col = i/3 ;

        auto xaxis = x0 + col * dxy;
        auto yaxis = y0 - line * dxy;
        auto arrowX = arrowX0 + col * dxy;
        auto arrowY = arrowY0 - line * dxy;

        auto arrowdir = (vertInds.at(i).dir() == In) ? "->" : "<-";

        psout << "\\psline[linecolor=colour10, linewidth=0.08, arrowsize=0.053cm 4.0,arrowlength=2.0,arrowinset=0.0]{"
             << arrowdir << "}(" << arrowX << "," << arrowY << ")("
             << arrowX << "," << arrowY - arrowLen << ")" << endl;

        psout << "\\rput[bl](" << xaxis
             << "," << yaxis << "){"
             << vertInds.at(i).id() % 1000 << "}" << endl;
    }

    psout<<"\\end{pspicture}"   <<endl;
    psout<<"}"<<endl;

    psout.clear();
}

#endif //IPEPS_PSTRICKCTM_H
