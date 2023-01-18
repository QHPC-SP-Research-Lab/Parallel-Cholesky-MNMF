#include "CPUdefines.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int   NChannels, NInverses, ndims, Typo, OmpCores, LibCores;
  const mwSize *dims;
 
  #ifdef TIMERS
    double Time;
  #endif

  if(nrhs != 6) mexErrMsgIdAndTxt("MATLAB:CPUSolver:nrhs", "6 parameters are needed");
  if(nlhs != 1) mexErrMsgIdAndTxt("MATLAB:CPUSolver:nhrs", "1 parameter  is returned");

  ndims=mxGetNumberOfDimensions(prhs[0]);
  if (ndims != 3) mexErrMsgIdAndTxt("MATLAB:CPUSolver:ndims", "3D matrices are needed");

  dims      = mxGetDimensions(prhs[0]);
  NChannels = dims[0];
  NInverses = dims[2];
  Typo      = (mwSize)mxGetScalar(prhs[3]);
  OmpCores  = (mwSize)mxGetScalar(prhs[4]);
  LibCores  = (mwSize)mxGetScalar(prhs[5]);

  if(Typo==1)
  {
    MyType *A=NULL, *B=NULL, *C=NULL, *R=NULL;

    A = mxGetDoubles(prhs[0]);
    B = mxGetDoubles(prhs[1]);
    C = mxGetDoubles(prhs[2]);

    if (A == NULL) mexErrMsgIdAndTxt("MATLAB:CPUSolver:NULL", "1st parameter is NULL.");
    if (B == NULL) mexErrMsgIdAndTxt("MATLAB:CPUSolver:NULL", "2nd parameter is NULL.");
    if (C == NULL) mexErrMsgIdAndTxt("MATLAB:CPUSolver:NULL", "3rd parameter is NULL.");

    plhs[0] = mxCreateDoubleMatrix(1, NInverses, mxREAL);
    R       = mxGetDoubles(plhs[0]);

    #ifdef TIMERS
      Time=dseconds();
    #endif
      RealCPU(NChannels, NInverses, A, B, C, R, OmpCores, LibCores);
    #ifdef TIMERS
      Time=dseconds()-Time;
    #endif
  }
  else
  {
    MyComplex *A=NULL, *B=NULL, *C=NULL, *R=NULL;

    A = (MyComplex *)mxGetComplexDoubles(prhs[0]);
    B = (MyComplex *)mxGetComplexDoubles(prhs[1]);
    C = (MyComplex *)mxGetComplexDoubles(prhs[2]);

    if (A == NULL) mexErrMsgIdAndTxt("MATLAB:CPUSolver:NULL", "1st parameter is NULL.");
    if (B == NULL) mexErrMsgIdAndTxt("MATLAB:CPUSolver:NULL", "2nd parameter is NULL.");
    if (C == NULL) mexErrMsgIdAndTxt("MATLAB:CPUSolver:NULL", "3rd parameter is NULL.");

    plhs[0]=mxCreateDoubleMatrix(1, NInverses, mxCOMPLEX);
    R      =(MyComplex *)mxGetComplexDoubles(plhs[0]);

    #ifdef TIMERS
      Time=dseconds();
    #endif
      ComplexCPU(NChannels, NInverses, A, B, C, R, OmpCores, LibCores);
    #ifdef TIMERS
      Time=dseconds()-Time;
    #endif
  }
  #ifdef TIMERS
    mexPrintf("CPU %f sec. per inverse\n", Time);
  #endif
} 
