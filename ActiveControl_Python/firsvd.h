#include <stdio.h>

class FIRFilter {

    private:
        int ptr = 0;

    public:
        float *wptr = 0; 
        float *x = 0;
        float y;
        int N = 5;
        FIRFilter();
        FIRFilter(int mem, float *ptrw, float *ptrx);        
        void setMem(int mem);
        void setMemAndPointers(int mem,float *ptrw, float *ptrx);
        void reset();
        float filter(float xn);

};


class FIRFilterSVD {

    private:
        int ptr = 0;
        FIRFilter firbranches[10] = {FIRFilter()};

    public:
        float *wptr;
        float *x;
        float y;
        int N = 5;
        int R = 10;
        int C = 10;
        int B = 1;
        float buffers[500] = {0};
        FIRFilterSVD();
        FIRFilterSVD(int mem, int nbranches, int nR, int nC, float *ptrw, float *ptrx);
        void setAllParams(int mem, int nbranches, int nR, int nC, float *ptrw, float *ptrx);
        void setParams(int mem, int nbranches, int nR, int nC);
        void reset();
        float filter(float xn);

};