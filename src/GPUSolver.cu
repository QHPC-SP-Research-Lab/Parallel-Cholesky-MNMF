#include "GPUdefines.cuh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int    NChannels, NInverses, ndims, deviceCount, error=0, Modo, Typo, Cores, Strea;
  const  mwSize *dims;
  cudaError_t cudaerr;

  #ifdef TIMERS
    float       Time;
    cudaEvent_t start, stop;
  #endif
  
  if(nrhs != 7) mexErrMsgIdAndTxt("MATLAB:GPUSolver:nrhs", "7 parameters are needed");
  if(nlhs != 1) mexErrMsgIdAndTxt("MATLAB:GPUSolver:nhrs", "1 parameter  is returned");

  ndims = mxGetNumberOfDimensions(prhs[0]);
  if(ndims != 3) mexErrMsgIdAndTxt("MATLAB:GPUSolver:ndims", "3D matrices are needed");

  dims      = mxGetDimensions(prhs[0]);
  NChannels = dims[0];
  NInverses = dims[2];
  Modo      = (mwSize)mxGetScalar(prhs[3]);
  Typo      = (mwSize)mxGetScalar(prhs[4]);
  Cores     = (mwSize)mxGetScalar(prhs[5]);
  Strea     = (mwSize)mxGetScalar(prhs[6]);

  cudaerr=cudaGetDeviceCount(&deviceCount);
  if((deviceCount < 1) || (cudaerr !=cudaSuccess))
    mexErrMsgIdAndTxt("MATLAB:GPUSolver:GPU","System has not compatible GPU\n");
  else
  {
    if(deviceCount > 1) mexPrintf("\nSystem has more than 1 compatible GPU. Device ID 0 will be used\n");

    #ifdef TIMERS
      cudaEventCreate (&start);
      cudaEventCreate (&stop);
    #endif

    if(Typo==1)
    { 
      MyType *A=NULL, *B=NULL, *C=NULL, *R=NULL;

      A = mxGetDoubles(prhs[0]);
      B = mxGetDoubles(prhs[1]);
      C = mxGetDoubles(prhs[2]);

      if (A == NULL) mexErrMsgIdAndTxt("MATLAB:GPUSolver:NULL", "1st parameter is not mxDOUBLE_CLASS array.");
      if (B == NULL) mexErrMsgIdAndTxt("MATLAB:GPUSolver:NULL", "2nd parameter is not mxDOUBLE_CLASS array.");
      if (C == NULL) mexErrMsgIdAndTxt("MATLAB:GPUSolver:NULL", "3rd parameter is not mxDOUBLE_CLASS array.");

      plhs[0] = mxCreateDoubleMatrix(1, NInverses, mxREAL);
      R       = mxGetDoubles(plhs[0]);

      cudaHostRegister(A, sizeof(MyType)*NChannels*NChannels*NInverses, cudaHostRegisterMapped);
      cudaHostRegister(B, sizeof(MyType)*NChannels*NChannels*NInverses, cudaHostRegisterMapped);
      cudaHostRegister(C, sizeof(MyType)*NChannels*NChannels*NInverses, cudaHostRegisterMapped);
      cudaHostRegister(R, sizeof(MyType)*NInverses,                     cudaHostRegisterMapped);

      #ifdef TIMERS
        cudaEventRecord(start, 0);
      #endif

      switch(Modo){
        case 1:
          error=RealStreams(NChannels, NInverses, A, B, C, R, Strea);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:RealStreams", "Problems with RealStreams\n");
          break;
        case 2:
          error=RealZeroCopy(NChannels, NInverses, A, B, C, R);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:RealZeroCopy", "Problems with RealZeroCopy\n");
          break;
        case 3:
          error=RealClassic(NChannels, NInverses, A, B, C, R);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:RealClassic", "Problems with RealClassic\n");
          break;
        case 4:
          error=RealOpenMP(NChannels, NInverses, A, B, C, R, Cores);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:RealOpenMP", "Problems with RealOpenMP\n");
          break;
        case 5:
          error=RealEvents(NChannels, NInverses, A, B, C, R);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:RealEvents", "Problems with RealEvents\n");
          break;
        case 6:
          error=RealEventsOMP(NChannels, NInverses, A, B, C, R, Cores);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:RealEventsOMP", "Problems with RealEventsOMP\n");
          break;
        default:
          mexErrMsgIdAndTxt("MATLAB:GPUSolver:NULL", "Unsupported mode\n");      
      }
      #ifdef TIMERS
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&Time, start, stop);
      #endif
    }
    else
    {
      MyComplex *A=NULL, *B=NULL, *C=NULL, *R=NULL;

      A=(MyComplex *)mxGetComplexDoubles(prhs[0]);
      B=(MyComplex *)mxGetComplexDoubles(prhs[1]);
      C=(MyComplex *)mxGetComplexDoubles(prhs[2]);

      if (A==NULL) mexErrMsgIdAndTxt("MATLAB:GPUSolver:NULL", "1st parameter is not mxComplexDouble_CLASS array.");
      if (B==NULL) mexErrMsgIdAndTxt("MATLAB:GPUSolver:NULL", "2nd parameter is not mxComplexDouble_CLASS array.");
      if (C==NULL) mexErrMsgIdAndTxt("MATLAB:GPUSolver:NULL", "3rd parameter is not mxComplexDouble_CLASS array.");

      plhs[0] = mxCreateDoubleMatrix(1, NInverses, mxCOMPLEX);
      R       = (MyComplex *)mxGetComplexDoubles(plhs[0]);

      cudaHostRegister(A, sizeof(MyComplex)*NChannels*NChannels*NInverses, cudaHostRegisterMapped);
      cudaHostRegister(B, sizeof(MyComplex)*NChannels*NChannels*NInverses, cudaHostRegisterMapped);
      cudaHostRegister(C, sizeof(MyComplex)*NChannels*NChannels*NInverses, cudaHostRegisterMapped);
      cudaHostRegister(R, sizeof(MyComplex)*NInverses,                     cudaHostRegisterMapped);

      #ifdef TIMERS
        cudaEventRecord(start, 0);
      #endif

      switch(Modo){
        case 1:
          error=ComplexStreams(NChannels, NInverses, A, B, C, R, Strea);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:ComplexStreams", "Problems with ComplexStreams\n");
          break;
        case 2:
          error=ComplexZeroCopy(NChannels, NInverses, A, B, C, R);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:ComplexZeroCopy", "Problems with ComplexZeroCopy\n");
          break;
        case 3:
          error=ComplexClassic(NChannels, NInverses, A, B, C, R);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:ComplexClassic", "Problems with ComplexClassic\n");
          break;
        case 4:
          error=ComplexOpenMP(NChannels, NInverses, A, B, C, R, Cores);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:ComplexOpenMP", "Problems with ComplexOpenMP\n");
          break;
        case 5:
          error=ComplexEvents(NChannels, NInverses, A, B, C, R);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:ComplexEvents", "Problems with ComplexEvents\n");
          break;
        case 6:
          error=ComplexEventsOMP(NChannels, NInverses, A, B, C, R, Cores);
          if (error != 0) mexErrMsgIdAndTxt("MATLAB:GPUSolver:ComplexEventsOMP", "Problems with ComplexEventsOMP\n");
          break;
        default:
          mexErrMsgIdAndTxt("MATLAB:GPUSolver:NULL", "Unsupported mode\n");
      }
      #ifdef TIMERS
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&Time, start, stop);
      #endif
    }
    #ifdef TIMERS
      mexPrintf("GPU %f sec. per inverse\n", Time/(1000.0*NInverses));
      cudaEventDestroy (start);
      cudaEventDestroy (stop);
    #endif
  }
}
