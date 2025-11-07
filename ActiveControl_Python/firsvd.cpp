// #include <stdio.h>
// #include <stdlib.h>
// #include <stdbool.h>
// #include <math.h>
// #include <string.h>
// #include <float.h>
#include "firsvd.h"

FIRFilter::FIRFilter() {
    wptr = 0;
    x = 0;
    N = 1;
}

FIRFilter::FIRFilter(int mem, float *ptrw, float *ptrx) {
    wptr = ptrw;
    x = ptrx;
    N = mem;
    reset();
}

void FIRFilter::setMem(int mem) {
    N = mem;
    reset();
}

void FIRFilter::setMemAndPointers(int mem, float *ptrw, float *ptrx) {
    wptr = ptrw;
    x = ptrx;
    N = mem;
    reset();
}

void FIRFilter::reset() {
    y = 0;
    for (int k = 0; k < N; k++) { *(x+k) = 0; }
    ptr = N-1;
}

float FIRFilter::filter(float xn) {
    ptr++;
    if (ptr >= N) { ptr = 0; }
    *(x+ptr) = xn;
    y = 0;
    for (int k = 0; k < N; k++) {
    y = y + *(x+((ptr-k+N)%N)) * *(wptr+k);
    //x[(ptr-k+N)%N] * *(wptr+k);
    }
    return y;
}

FIRFilterSVD::FIRFilterSVD() {
    wptr = nullptr;
    x = nullptr;
    N = 0;
    R = 0;
    C = 0;
    B = 0;
    ptr = 0;
}

FIRFilterSVD::FIRFilterSVD(int mem, int nbranches, int nR, int nC, float *ptrw, float *ptrx) {
    wptr = ptrw;
    x = ptrx;
    N = mem;
    R = nR;
    C = nC;
    B = nbranches;
    reset();
}

void FIRFilterSVD::setAllParams(int mem, int nbranches, int nR, int nC, float *ptrw, float *ptrx) {
    wptr = ptrw;
    x = ptrx;
    N = mem;
    R = nR;
    C = nC;
    B = nbranches;
    reset();
}

void FIRFilterSVD::setParams(int mem, int nbranches, int nR, int nC) {
    N = mem;
    R = nR;
    C = nC;
    B = nbranches;
    reset();
}

void FIRFilterSVD::reset() {
    y = 0;
    for (int k = 0; k < N; k++) { *(x+k) = 0; }
    ptr = N-1;
    for (int k = 0; k < B; k++) {
        firbranches[k].setMemAndPointers(R, wptr + B*C + k*R, buffers + k*R);
    }
}

float FIRFilterSVD::filter(float xn) {
    ptr++;
    if (ptr >= N) { ptr = 0; }
    *(x+ptr) = xn;
    y = 0;
    for (int bb = 0; bb < B; bb++) {
        float aux = 0.0;
        float * wcptr = wptr + bb*C;
        for (int cc = 0; cc < C; cc++) {
            aux = aux + *(x+((ptr-(cc*R)+N)%N)) * *(wcptr++);
        }
        y = y + firbranches[bb].filter(aux);
    }
    return y;
}

extern "C" {

    FIRFilterSVD myfilter;

    FIRFilter myfirfilter;

    void createFilter(int mem, int nbranches, int nR, int nC, float *ptrw, float *ptrx) {
        myfilter.setAllParams(mem, nbranches, nR, nC, ptrw, ptrx);
        myfirfilter.setMemAndPointers(mem, ptrw, ptrx);
    }

    float Filter(float xn) {
        return myfilter.filter(xn);        
    }

    float FilterWithFIR(float xn) {
        return myfirfilter.filter(xn);        
    }
}