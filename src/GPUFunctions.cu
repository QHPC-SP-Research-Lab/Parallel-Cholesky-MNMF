#include "GPUdefines.cuh"


__global__ void cFinalTraceForCublas(const unsigned int n, const MyComplex Num, MyComplex *X)
{
  // For only one block with 32 threads where n >= 32 //
  MyComplex ctmp;

  ctmp.x=ctmp.y=0.0;

  for(unsigned int i=0; i<n; i+=32)
    if ((threadIdx.x+i) < n)
      ctmp=cuCadd(ctmp, X[(threadIdx.x+i)*n+threadIdx.x+i]);

  ctmp.x=ReduceWarp(ctmp.x);
  ctmp.y=ReduceWarp(ctmp.y);

  if (threadIdx.x == 0)
    X[0]=cuCdiv(Num, ctmp);
}


__global__ void dFinalTraceForCublas(const unsigned int n, const MyType Num, MyType *X)
{
  // For only one block with 32 threads where n >= 32 //
  MyType dtmp=0.0;
  for(unsigned int i=0; i<n; i+=32)
    if ((threadIdx.x+i) < n)
      dtmp += X[(threadIdx.x+i)*n+threadIdx.x+i];
  dtmp=ReduceWarp(dtmp);
  if (threadIdx.x == 0)
    X[0]=Num/dtmp;
}


__global__ void cFinalTraceForKernel(const unsigned int n, const MyComplex *B, MyComplex *C, MyComplex *R)
{
  // For only one block with 32 threads where n >= 32 //
  MyComplex nume, deno;
  
  nume.x=nume.y=deno.x=deno.y=0.0;
  
  for(unsigned int i=0; i<n; i+=32)
    if ((threadIdx.x+i) < n)
    {
      nume=cuCadd(nume, B[threadIdx.x+i]);
      deno=cuCadd(deno, C[(threadIdx.x+i)*n+threadIdx.x+i]);
    }

  nume.x=ReduceWarp(nume.x); nume.y=ReduceWarp(nume.y); 
  deno.x=ReduceWarp(deno.x); deno.y=ReduceWarp(deno.y);

  if (threadIdx.x == 0)
    R[0]=cuCdiv(nume, deno);
}


__global__ void dFinalTraceForKernel(const unsigned int n, const MyType *B, MyType *C, MyType *R)
{
  // For only one block with 32 threads where n >= 32 //
  MyType nume=0.0, deno=0.0;
  for(unsigned int i=0; i<n; i+=32)
  {
    if ((threadIdx.x+i) < n)
    {
      nume += B[threadIdx.x+i];
      deno += C[(threadIdx.x+i)*n+threadIdx.x+i];
    }
  }
  nume=ReduceWarp(nume);
  deno=ReduceWarp(deno);

  if (threadIdx.x == 0)
    R[0]=nume/deno;
}


__global__ void cTraceHalfCoale(const unsigned int n, MyComplex *A, const MyComplex *B)
{
  if (blockIdx.x < n)
  {
    unsigned int yo=(blockIdx.x * n + threadIdx.x), pos=threadIdx.x*n+blockIdx.x, i;
    MyComplex ctmp;

    ctmp.x=ctmp.y=0.0;

    for(i=0; i<n; i+=32)
    {
      if (threadIdx.x+i < n)
        // cuCfma(x,y,z) --> x*y+z
        ctmp=cuCfma(A[i*n+pos], B[yo+i], ctmp);
    }

    ctmp.x=ReduceWarp(ctmp.x);
    ctmp.y=ReduceWarp(ctmp.y);

    if (threadIdx.x == 0)
      A[blockIdx.x]=ctmp;
  }
}


__global__ void dTraceHalfCoale(const unsigned int n, MyType *A, const MyType *B)
{
  if (blockIdx.x < n)
  {
    unsigned int yo=(blockIdx.x * n + threadIdx.x), pos=threadIdx.x*n+blockIdx.x, i;
    MyType dtmp= 0.0;
    for(i=0; i<n; i+=32)
    {
      if (threadIdx.x+i < n)
        dtmp += A[i*n+pos]*B[yo+i];
    }
    dtmp=ReduceWarp(dtmp);
    if (threadIdx.x == 0) A[blockIdx.x]=dtmp;
  }
}


int ComplexOneClassicStep(cusolverDnHandle_t handle, const int NChan, const int SizeTRF, int *Info, MyComplex *U, MyComplex *X, MyComplex *Y, MyComplex *TRF, MyComplex *R)
{
  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnCpotrf(handle, CUBLAS_FILL_MODE_UPPER, NChan, U, NChan, TRF,   SizeTRF,  Info));
    CUSOLVERERR(cusolverDnCpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, X, NChan, Info));
    CUSOLVERERR(cusolverDnCpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, Y, NChan, Info));
  #else
    CUSOLVERERR(cusolverDnZpotrf(handle, CUBLAS_FILL_MODE_UPPER, NChan, U, NChan, TRF,   SizeTRF,  Info));
    CUSOLVERERR(cusolverDnZpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, X, NChan, Info));
    CUSOLVERERR(cusolverDnZpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, Y, NChan, Info));
  #endif

  cTraceHalfCoale<<<NChan,  32>>>(NChan, X, Y); 
  cFinalTraceForKernel<<<1, 32>>>(NChan, X, Y, R);

  /* With CUBLAS 
  MyComplex Num, alfa, beta;
  alfa.x=1.0; alfa.y=0.0;
  beta.x=0.0; beta.y=0.0;

  // Compute Num=Trace[(U^-1 * X) * (U^-1 * Y)]. (U^-1 * X) es X y (U^-1 * Y) es Y.
  #ifdef SIMPLE
    // U = X^t
    CUBLASERR(cublasCgeam(handleC, CUBLAS_OP_T, CUBLAS_OP_N, NChannels, NChannels, &alfa, X, NChannels, &beta, NULL, NChannels, U, NChannels));

    // Num=X^t*Y
    CUBLASERR(cublasCdotu(handleC, NChannels*NChannels, U, 1, Y, 1, &Num));
  #else
    // U = X^t
    CUBLASERR(cublasZgeam(handleC, CUBLAS_OP_T, CUBLAS_OP_N, NChannels, NChannels, &alfa, X, NChannels, &beta, NULL, NChannels, U, NChannels));

    // Num=X^t*Y
    CUBLASERR(cublasZdotu(handleC, NChannels*NChannels, U, 1, Y, 1, &Num));
  #endif
  
  // Compute Y=(Num / Trace[(U^-1 * Y)]). (U^-1 * Y). The result is in Y[0]
  cFinalTraceForCublas<<<1, 32>>>(NChannels, Num, Y);
  */

  return 0;
}


int RealOneClassicStep(cusolverDnHandle_t handle, const int NChan, const int SizeTRF, int *Info, MyType *U, MyType *X, MyType *Y, MyType *TRF, MyType *R)
{
  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_UPPER, NChan, U, NChan, TRF,   SizeTRF,  Info));
    CUSOLVERERR(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, X, NChan, Info));
    CUSOLVERERR(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, Y, NChan, Info));
  #else
    CUSOLVERERR(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_UPPER, NChan, U, NChan, TRF,   SizeTRF,  Info));
    CUSOLVERERR(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, X, NChan, Info));
    CUSOLVERERR(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, Y, NChan, Info));
  #endif

  dTraceHalfCoale<<<NChan,  32>>>(NChan, X, Y); 
  dFinalTraceForKernel<<<1, 32>>>(NChan, X, Y, R);
  
  return 0;
}


int ComplexOneStreamStep(cusolverDnHandle_t handle, cudaStream_t stream, const int NChan, const int SizeTRF, int *Info, MyComplex *U, MyComplex *X, MyComplex *Y, MyComplex *TRF, MyComplex *R)
{
  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnCpotrf(handle, CUBLAS_FILL_MODE_UPPER, NChan, U, NChan, TRF,   SizeTRF,  Info));
    CUSOLVERERR(cusolverDnCpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, X, NChan, Info));
    CUSOLVERERR(cusolverDnCpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, Y, NChan, Info));
  #else
    CUSOLVERERR(cusolverDnZpotrf(handle, CUBLAS_FILL_MODE_UPPER, NChan, U, NChan, TRF,   SizeTRF,  Info));
    CUSOLVERERR(cusolverDnZpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, X, NChan, Info));
    CUSOLVERERR(cusolverDnZpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, Y, NChan, Info));
  #endif

  cTraceHalfCoale<<<NChan,  32, 0, stream>>>(NChan, X, Y); 
  cFinalTraceForKernel<<<1, 32, 0, stream>>>(NChan, X, Y, R);
  
  return 0;
}


int RealOneStreamStep(cusolverDnHandle_t handle, cudaStream_t stream, const int NChan, const int SizeTRF, int *Info, MyType *U, MyType *X, MyType *Y, MyType *TRF, MyType *R)
{
  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_UPPER, NChan, U, NChan, TRF,   SizeTRF,  Info));
    CUSOLVERERR(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, X, NChan, Info));
    CUSOLVERERR(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, Y, NChan, Info));
  #else
    CUSOLVERERR(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_UPPER, NChan, U, NChan, TRF,   SizeTRF,  Info));
    CUSOLVERERR(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, X, NChan, Info));
    CUSOLVERERR(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_UPPER, NChan, NChan, U, NChan, Y, NChan, Info));
  #endif

  dTraceHalfCoale<<<NChan,  32, 0, stream>>>(NChan, X, Y); 
  dFinalTraceForKernel<<<1, 32, 0, stream>>>(NChan, X, Y, R);
  
  return 0;
}


int ComplexClassic(const int NChannels, const int NInverses, const MyComplex *A, const MyComplex *B, const MyComplex *C, MyComplex *Result)
{
  size_t     Size;
  int       *Err=NULL, SizeTRF, i;
  MyComplex *U=NULL, *X=NULL, *Y=NULL, *R=NULL, *TRF=NULL;

  cusolverDnHandle_t handle=NULL;

  Size=(size_t)NChannels * (size_t)NChannels;

  CUSOLVERERR(cusolverDnCreate(&handle));

  CUDAERR(cudaMalloc((void **)&U, sizeof(MyComplex)*Size));
  CUDAERR(cudaMalloc((void **)&X, sizeof(MyComplex)*Size));
  CUDAERR(cudaMalloc((void **)&Y, sizeof(MyComplex)*Size));
  CUDAERR(cudaMalloc((void **)&R, sizeof(MyComplex)*NInverses));

  CUDAERR(cudaMemcpy(U, A, sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(X, B, sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Y, C, sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));

  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnCpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #else
    CUSOLVERERR(cusolverDnZpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #endif
  CUDAERR(cudaMalloc((void **)&TRF, sizeof(MyComplex)*SizeTRF));
  CUDAERR(cudaMalloc((void **)&Err, sizeof(int)));

  ComplexOneClassicStep(handle, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[0]);

  for(i=1; i<NInverses; i++)
  {
    CUDAERR(cudaMemcpy(U, &A[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
    CUDAERR(cudaMemcpy(X, &B[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
    CUDAERR(cudaMemcpy(Y, &C[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));

    ComplexOneClassicStep(handle, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[i]);
  }
  CUDAERR(cudaMemcpy(Result, R, sizeof(MyComplex)*NInverses, cudaMemcpyDeviceToHost));

  CUDAERR(cudaFree(TRF)); CUDAERR(cudaFree(Err)); CUDAERR(cudaFree(Y));
  CUDAERR(cudaFree(U));   CUDAERR(cudaFree(X));   CUDAERR(cudaFree(R));

  CUSOLVERERR(cusolverDnDestroy(handle));
  
  return 0;
}


int RealClassic(const int NChannels, const int NInverses, const MyType *A, const MyType *B, const MyType *C, MyType *Result)
{
  size_t  Size;
  int     *Err=NULL, SizeTRF, i;
  MyType  *U=NULL, *X=NULL, *Y=NULL, *R=NULL, *TRF=NULL;

  cusolverDnHandle_t handle=NULL;

  Size=(size_t)NChannels * (size_t)NChannels;

  CUSOLVERERR(cusolverDnCreate(&handle));

  CUDAERR(cudaMalloc((void **)&U, sizeof(MyType)*Size));
  CUDAERR(cudaMalloc((void **)&X, sizeof(MyType)*Size));
  CUDAERR(cudaMalloc((void **)&Y, sizeof(MyType)*Size));
  CUDAERR(cudaMalloc((void **)&R, sizeof(MyType)*NInverses));

  CUDAERR(cudaMemcpy(U, A, sizeof(MyType)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(X, B, sizeof(MyType)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Y, C, sizeof(MyType)*Size, cudaMemcpyHostToDevice));

  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #else
    CUSOLVERERR(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #endif
  CUDAERR(cudaMalloc((void **)&TRF, sizeof(MyType)*SizeTRF));
  CUDAERR(cudaMalloc((void **)&Err, sizeof(int)));

  RealOneClassicStep(handle, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[0]);

  for(i=1; i<NInverses; i++)
  {
    CUDAERR(cudaMemcpy(U, &A[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice));
    CUDAERR(cudaMemcpy(X, &B[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice));
    CUDAERR(cudaMemcpy(Y, &C[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice));

    RealOneClassicStep(handle, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[i]);
  }
  CUDAERR(cudaMemcpy(Result, R, sizeof(MyType)*NInverses, cudaMemcpyDeviceToHost));

  CUDAERR(cudaFree(TRF)); CUDAERR(cudaFree(Err)); CUDAERR(cudaFree(Y));
  CUDAERR(cudaFree(U));   CUDAERR(cudaFree(X));   CUDAERR(cudaFree(R));

  CUSOLVERERR(cusolverDnDestroy(handle));
  
  return 0;
}


int ComplexZeroCopy(const int NChannels, const int NInverses, MyComplex *A, MyComplex *B, MyComplex *C, MyComplex *Result)
{
  size_t    Size;
  int       *Err=NULL, SizeTRF, i;
  MyComplex *U=NULL, *X=NULL, *Y=NULL, *R=NULL, *TRF=NULL;

  cusolverDnHandle_t handle=NULL;

  Size=(size_t)NChannels * (size_t)NChannels;

  CUDAERR(cudaSetDeviceFlags(cudaDeviceMapHost));

  CUSOLVERERR(cusolverDnCreate(&handle));

  CUDAERR(cudaHostGetDevicePointer((void **)&U, (void *)A, 0));
  CUDAERR(cudaHostGetDevicePointer((void **)&X, (void *)B, 0));
  CUDAERR(cudaHostGetDevicePointer((void **)&Y, (void *)C, 0));

  CUDAERR(cudaMalloc((void **)&R, sizeof(MyComplex)*NInverses));

  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnCpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #else
    CUSOLVERERR(cusolverDnZpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #endif
  CUDAERR(cudaMalloc((void **)&TRF, sizeof(MyComplex)*SizeTRF));
  CUDAERR(cudaMalloc((void **)&Err, sizeof(int)));

  ComplexOneClassicStep(handle, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[0]);

  for(i=1; i<NInverses; i++)
    ComplexOneClassicStep(handle, NChannels, SizeTRF, Err, &U[Size*i], &X[Size*i], &Y[Size*i], TRF, &R[i]);

  CUDAERR(cudaMemcpy(Result, R, sizeof(MyComplex)*NInverses, cudaMemcpyDeviceToHost));

  CUDAERR(cudaFree(TRF)); CUDAERR(cudaFree(Err)); CUDAERR(cudaFree(R));

  CUSOLVERERR(cusolverDnDestroy(handle));
  
  return 0;
}


int RealZeroCopy(const int NChannels, const int NInverses, MyType *A, MyType *B, MyType *C, MyType *Result)
{
  size_t  Size;
  int     *Err=NULL, SizeTRF, i;
  MyType  *U=NULL, *X=NULL, *Y=NULL, *R=NULL, *TRF=NULL;

  cusolverDnHandle_t handle=NULL;

  Size=(size_t)NChannels * (size_t)NChannels;

  CUDAERR(cudaSetDeviceFlags(cudaDeviceMapHost));

  CUSOLVERERR(cusolverDnCreate(&handle));

  CUDAERR(cudaHostGetDevicePointer((void **)&U, (void *)A, 0));
  CUDAERR(cudaHostGetDevicePointer((void **)&X, (void *)B, 0));
  CUDAERR(cudaHostGetDevicePointer((void **)&Y, (void *)C, 0));

  CUDAERR(cudaMalloc((void **)&R, sizeof(MyType)*NInverses));

  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #else
    CUSOLVERERR(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #endif
  CUDAERR(cudaMalloc((void **)&TRF, sizeof(MyType)*SizeTRF));
  CUDAERR(cudaMalloc((void **)&Err, sizeof(int)));

  RealOneClassicStep(handle, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[0]);

  for(i=1; i<NInverses; i++)
    RealOneClassicStep(handle, NChannels, SizeTRF, Err, &U[Size*i], &X[Size*i], &Y[Size*i], TRF, &R[i]);

  CUDAERR(cudaMemcpy(Result, R, sizeof(MyType)*NInverses, cudaMemcpyDeviceToHost));

  CUDAERR(cudaFree(TRF)); CUDAERR(cudaFree(Err)); CUDAERR(cudaFree(R));

  CUSOLVERERR(cusolverDnDestroy(handle));
  
  return 0;
}


int ComplexOpenMP(const int NChannels, const int NInverses, const MyComplex *A, const MyComplex *B, const MyComplex *C, MyComplex *Result, const int Cores)
{
  MyComplex *R=NULL;

  CUDAERR(cudaMalloc((void **)&R, sizeof(MyComplex)*NInverses));

  #ifdef OMP
    #pragma omp parallel num_threads(Cores)
  {
  #endif
    size_t    Size;
    int       *Err=NULL, SizeTRF, i, Chunk, MyID=0, NumTh=1, Start, End;
    MyComplex *U=NULL, *X=NULL, *Y=NULL, *TRF=NULL;

    cusolverDnHandle_t handle;
    cudaStream_t       stream;

    Size=(size_t)NChannels * (size_t)NChannels;

    CUSOLVERERROMP(cusolverDnCreate(&handle));
    CUDAERROMP(cudaStreamCreate(&stream));
    CUSOLVERERROMP(cusolverDnSetStream(handle, stream));
    
    #ifdef OMP
      MyID  = omp_get_thread_num();
      NumTh = omp_get_num_threads();
    #endif
    Chunk = NInverses / NumTh;
    Start = MyID * Chunk;

    if (MyID==(NumTh-1)) { End=Start + Chunk + (NInverses % NumTh); } else { End=Start + Chunk; }

    CUDAERROMP(cudaMalloc((void **)&U, sizeof(MyComplex)*Size));
    CUDAERROMP(cudaMalloc((void **)&X, sizeof(MyComplex)*Size));
    CUDAERROMP(cudaMalloc((void **)&Y, sizeof(MyComplex)*Size));

    CUDAERROMP(cudaMemcpyAsync(U, &A[Start*Size], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream));
    CUDAERROMP(cudaMemcpyAsync(X, &B[Start*Size], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream));
    CUDAERROMP(cudaMemcpyAsync(Y, &C[Start*Size], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream));

    #ifdef SIMPLE
      CUSOLVERERROMP(cusolverDnCpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
    #else
      CUSOLVERERROMP(cusolverDnZpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
    #endif
    CUDAERROMP(cudaMalloc((void **)&TRF, sizeof(MyComplex)*SizeTRF));
    CUDAERROMP(cudaMalloc((void **)&Err, sizeof(int)));

    ComplexOneStreamStep(handle, stream, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[Start]);

    for(i=(Start+1); i<End; i++)
    {
      CUDAERROMP(cudaMemcpyAsync(U, &A[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream));
      CUDAERROMP(cudaMemcpyAsync(X, &B[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream));
      CUDAERROMP(cudaMemcpyAsync(Y, &C[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream));

      ComplexOneStreamStep(handle, stream, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[i]);
    }
    CUDAERROMP(cudaStreamSynchronize(stream));
    CUSOLVERERROMP(cusolverDnDestroy(handle));
    CUDAERROMP(cudaStreamDestroy(stream));

    CUDAERROMP(cudaFree(TRF)); CUDAERROMP(cudaFree(Err)); CUDAERROMP(cudaFree(Y));
    CUDAERROMP(cudaFree(U));   CUDAERROMP(cudaFree(X));
  #ifdef OMP
  }
  #endif  
  CUDAERROMP(cudaMemcpy(Result, R, sizeof(MyComplex)*NInverses, cudaMemcpyDeviceToHost));
  CUDAERROMP(cudaFree(R)); 

  return 0;
}


int RealOpenMP(const int NChannels, const int NInverses, const MyType *A, const MyType *B, const MyType *C, MyType *Result, const int Cores)
{
  MyType *R=NULL;

  CUDAERR(cudaMalloc((void **)&R, sizeof(MyType)*NInverses));

  #ifdef OMP
    #pragma omp parallel num_threads(Cores)
  {
  #endif
    size_t  Size;
    int     *Err=NULL, SizeTRF, i, Chunk, MyID=0, NumTh=1, Start, End;
    MyType  *U=NULL, *X=NULL, *Y=NULL, *TRF=NULL;

    cusolverDnHandle_t handle;
    cudaStream_t       stream;

    Size=(size_t)NChannels * (size_t)NChannels;

    CUSOLVERERROMP(cusolverDnCreate(&handle));
    CUDAERROMP(cudaStreamCreate(&stream));
    CUSOLVERERROMP(cusolverDnSetStream(handle, stream));
    
    #ifdef OMP
      MyID  = omp_get_thread_num();
      NumTh = omp_get_num_threads();
    #endif
    Chunk = NInverses / NumTh;
    Start = MyID * Chunk;

    if (MyID==(NumTh-1)) { End=Start + Chunk + (NInverses % NumTh); } else { End=Start + Chunk; }

    CUDAERROMP(cudaMalloc((void **)&U, sizeof(MyType)*Size));
    CUDAERROMP(cudaMalloc((void **)&X, sizeof(MyType)*Size));
    CUDAERROMP(cudaMalloc((void **)&Y, sizeof(MyType)*Size));

    CUDAERROMP(cudaMemcpyAsync(U, &A[Start*Size], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream));
    CUDAERROMP(cudaMemcpyAsync(X, &B[Start*Size], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream));
    CUDAERROMP(cudaMemcpyAsync(Y, &C[Start*Size], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream));

    #ifdef SIMPLE
      CUSOLVERERROMP(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
    #else
      CUSOLVERERROMP(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
    #endif
    CUDAERROMP(cudaMalloc((void **)&TRF, sizeof(MyType)*SizeTRF));
    CUDAERROMP(cudaMalloc((void **)&Err, sizeof(int)));

    RealOneStreamStep(handle, stream, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[Start]);

    for(i=(Start+1); i<End; i++)
    {
      CUDAERROMP(cudaMemcpyAsync(U, &A[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream));
      CUDAERROMP(cudaMemcpyAsync(X, &B[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream));
      CUDAERROMP(cudaMemcpyAsync(Y, &C[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream));

      RealOneStreamStep(handle, stream, NChannels, SizeTRF, Err, U, X, Y, TRF, &R[i]);
    }
    CUDAERROMP(cudaStreamSynchronize(stream));
    CUSOLVERERROMP(cusolverDnDestroy(handle));
    CUDAERROMP(cudaStreamDestroy(stream));

    CUDAERROMP(cudaFree(TRF)); CUDAERROMP(cudaFree(Err)); CUDAERROMP(cudaFree(Y));
    CUDAERROMP(cudaFree(U));   CUDAERROMP(cudaFree(X));
  #ifdef OMP
  }
  #endif  
  CUDAERROMP(cudaMemcpy(Result, R, sizeof(MyType)*NInverses, cudaMemcpyDeviceToHost));
  CUDAERROMP(cudaFree(R)); 

  return 0;
}


int ComplexStreams(const int NChannels, const int NInverses, const MyComplex *A, const MyComplex *B, const MyComplex *C, MyComplex *Result, const int NStreams)
{
  size_t    Size;
  int       *Err=NULL, SizeTRF, i, j;
  MyComplex *U=NULL, *X=NULL, *Y=NULL, *R=NULL, *TRF=NULL;

  cusolverDnHandle_t handles[NStreams];
  cudaStream_t       streams[NStreams];

  for (i=0; i<NStreams; i++)
  {
    CUDAERR(cudaStreamCreate(&streams[i]));
    CUSOLVERERR(cusolverDnCreate(&handles[i]));
    CUSOLVERERR(cusolverDnSetStream(handles[i], streams[i]));
  }  
  Size=(size_t)NChannels * (size_t)NChannels;

  CUDAERR(cudaMalloc((void **)&U, sizeof(MyComplex)*Size*NStreams));
  CUDAERR(cudaMalloc((void **)&X, sizeof(MyComplex)*Size*NStreams));
  CUDAERR(cudaMalloc((void **)&Y, sizeof(MyComplex)*Size*NStreams));
  CUDAERR(cudaMalloc((void **)&R, sizeof(MyComplex)*NInverses));

  CUDAERR(cudaMemcpy(U, A, sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(X, B, sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Y, C, sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnCpotrf_bufferSize(handles[0], CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #else
    CUSOLVERERR(cusolverDnZpotrf_bufferSize(handles[0], CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #endif
  CUDAERR(cudaMalloc((void **)&TRF, sizeof(MyComplex)*SizeTRF*NStreams));
  CUDAERR(cudaMalloc((void **)&Err, sizeof(int)*NStreams));

  for(i=0; i<(NInverses-NStreams); i+=NStreams)
  {
    for(j=0; j<NStreams; j++)
    {
      CUDAERR(cudaMemcpyAsync(&U[Size*j], &A[Size*(i+j)], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, streams[j]));
      CUDAERR(cudaMemcpyAsync(&X[Size*j], &B[Size*(i+j)], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, streams[j]));
      CUDAERR(cudaMemcpyAsync(&Y[Size*j], &C[Size*(i+j)], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, streams[j]));

      ComplexOneStreamStep(handles[j], streams[j], NChannels, SizeTRF, &Err[j], &U[Size*j], &X[Size*j], &Y[Size*j], &TRF[SizeTRF*j], &R[i+j]);
    }
  }
  for (j=0; j<(NInverses-i); j++)
  {
    CUDAERR(cudaMemcpyAsync(&U[Size*j], &A[Size*(i+j)], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, streams[j]));
    CUDAERR(cudaMemcpyAsync(&X[Size*j], &B[Size*(i+j)], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, streams[j]));
    CUDAERR(cudaMemcpyAsync(&Y[Size*j], &C[Size*(i+j)], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, streams[j]));

    ComplexOneStreamStep(handles[j], streams[j], NChannels, SizeTRF, &Err[j], &U[Size*j], &X[Size*j], &Y[Size*j], &TRF[SizeTRF*j], &R[i+j]);
  }
  for (i=0; i<NStreams; i++)
    CUDAERR(cudaStreamSynchronize(streams[i]));
  CUDAERR(cudaMemcpy(Result, R, sizeof(MyComplex)*NInverses, cudaMemcpyDeviceToHost));
  for (i=0; i<NStreams; i++)
  {
    CUSOLVERERR(cusolverDnDestroy(handles[i]));
    CUDAERR(cudaStreamDestroy(streams[i]));
  }

  CUDAERR(cudaFree(TRF)); CUDAERR(cudaFree(Err)); CUDAERR(cudaFree(Y));
  CUDAERR(cudaFree(U));   CUDAERR(cudaFree(X));   CUDAERR(cudaFree(R));

  return 0;
}


int RealStreams(const int NChannels, const int NInverses, const MyType *A, const MyType *B, const MyType *C, MyType *Result, const int NStreams)
{
  size_t  Size;
  int     *Err=NULL, SizeTRF, i, j;
  MyType  *U=NULL, *X=NULL, *Y=NULL, *R=NULL, *TRF=NULL;

  cusolverDnHandle_t handles[NStreams];
  cudaStream_t       streams[NStreams];

  for (i=0; i<NStreams; i++)
  {
    CUDAERR(cudaStreamCreate(&streams[i]));
    CUSOLVERERR(cusolverDnCreate(&handles[i]));
    CUSOLVERERR(cusolverDnSetStream(handles[i], streams[i]));
  }  
  Size=(size_t)NChannels * (size_t)NChannels;

  CUDAERR(cudaMalloc((void **)&U, sizeof(MyType)*Size*NStreams));
  CUDAERR(cudaMalloc((void **)&X, sizeof(MyType)*Size*NStreams));
  CUDAERR(cudaMalloc((void **)&Y, sizeof(MyType)*Size*NStreams));
  CUDAERR(cudaMalloc((void **)&R, sizeof(MyType)*NInverses));

  CUDAERR(cudaMemcpy(U, A, sizeof(MyType)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(X, B, sizeof(MyType)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Y, C, sizeof(MyType)*Size, cudaMemcpyHostToDevice));
  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnSpotrf_bufferSize(handles[0], CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #else
    CUSOLVERERR(cusolverDnDpotrf_bufferSize(handles[0], CUBLAS_FILL_MODE_UPPER, NChannels, U, NChannels, &SizeTRF));
  #endif
  CUDAERR(cudaMalloc((void **)&TRF, sizeof(MyType)*SizeTRF*NStreams));
  CUDAERR(cudaMalloc((void **)&Err, sizeof(int)*NStreams));

  for(i=0; i<(NInverses-NStreams); i+=NStreams)
  {
    for(j=0; j<NStreams; j++)
    {
      CUDAERR(cudaMemcpyAsync(&U[Size*j], &A[Size*(i+j)], sizeof(MyType)*Size, cudaMemcpyHostToDevice, streams[j]));
      CUDAERR(cudaMemcpyAsync(&X[Size*j], &B[Size*(i+j)], sizeof(MyType)*Size, cudaMemcpyHostToDevice, streams[j]));
      CUDAERR(cudaMemcpyAsync(&Y[Size*j], &C[Size*(i+j)], sizeof(MyType)*Size, cudaMemcpyHostToDevice, streams[j]));

      RealOneStreamStep(handles[j], streams[j], NChannels, SizeTRF, &Err[j], &U[Size*j], &X[Size*j], &Y[Size*j], &TRF[SizeTRF*j], &R[i+j]);
    }
  }
  for (j=0; j<(NInverses-i); j++)
  {
    CUDAERR(cudaMemcpyAsync(&U[Size*j], &A[Size*(i+j)], sizeof(MyType)*Size, cudaMemcpyHostToDevice, streams[j]));
    CUDAERR(cudaMemcpyAsync(&X[Size*j], &B[Size*(i+j)], sizeof(MyType)*Size, cudaMemcpyHostToDevice, streams[j]));
    CUDAERR(cudaMemcpyAsync(&Y[Size*j], &C[Size*(i+j)], sizeof(MyType)*Size, cudaMemcpyHostToDevice, streams[j]));

    RealOneStreamStep(handles[j], streams[j], NChannels, SizeTRF, &Err[j], &U[Size*j], &X[Size*j], &Y[Size*j], &TRF[SizeTRF*j], &R[i+j]);
  }
  for (i=0; i<NStreams; i++)
    CUDAERR(cudaStreamSynchronize(streams[i]));
  CUDAERR(cudaMemcpy(Result, R, sizeof(MyType)*NInverses, cudaMemcpyDeviceToHost));
  for (i=0; i<NStreams; i++)
  {
    CUSOLVERERR(cusolverDnDestroy(handles[i]));
    CUDAERR(cudaStreamDestroy(streams[i]));
  }

  CUDAERR(cudaFree(TRF)); CUDAERR(cudaFree(Err)); CUDAERR(cudaFree(Y));
  CUDAERR(cudaFree(U));   CUDAERR(cudaFree(X));   CUDAERR(cudaFree(R));

  return 0;
}


int ComplexEvents(const int NChannels, const int NInverses, const MyComplex *A, const MyComplex *B, const MyComplex *C, MyComplex *Result)
{
  size_t     Size;
  int        *Err=NULL, SizeTRF, i;
  MyComplex  *U1=NULL, *X1=NULL, *Y1=NULL, *U2=NULL, *X2=NULL, *Y2=NULL, *R=NULL, *TRF=NULL;

  cusolverDnHandle_t handle;
  cudaStream_t       stream1, stream2;
  cudaEvent_t        evento1, evento2;

  CUDAERR(cudaStreamCreate(&stream1));
  CUDAERR(cudaStreamCreate(&stream2));
  CUDAERR(cudaEventCreate(&evento1));
  CUDAERR(cudaEventCreate(&evento2));
  CUSOLVERERR(cusolverDnCreate(&handle));
  CUSOLVERERR(cusolverDnSetStream(handle, stream2));

  Size=(size_t)NChannels * (size_t)NChannels;

  CUDAERR(cudaMalloc((void **)&U1, sizeof(MyComplex)*Size));
  CUDAERR(cudaMalloc((void **)&X1, sizeof(MyComplex)*Size));
  CUDAERR(cudaMalloc((void **)&Y1, sizeof(MyComplex)*Size));
  CUDAERR(cudaMalloc((void **)&U2, sizeof(MyComplex)*Size));
  CUDAERR(cudaMalloc((void **)&X2, sizeof(MyComplex)*Size));
  CUDAERR(cudaMalloc((void **)&Y2, sizeof(MyComplex)*Size));
  CUDAERR(cudaMalloc((void **)&R,  sizeof(MyComplex)*NInverses));

  CUDAERR(cudaMemcpy(U1, A, sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(X1, B, sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Y1, C, sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnCpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U1, NChannels, &SizeTRF));
  #else
    CUSOLVERERR(cusolverDnZpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U1, NChannels, &SizeTRF));
  #endif
  CUDAERR(cudaMalloc((void **)&TRF, sizeof(MyComplex)*SizeTRF));
  CUDAERR(cudaMalloc((void **)&Err, sizeof(int)));

  cudaEventRecord(evento1, stream1);
  for (i=1; i<NInverses; i++)
  {
    if ((i % 2) != 0)
    {
      cudaStreamWaitEvent(stream2, evento1);
      ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[i-1]);
      cudaEventRecord(evento2, stream2);

      CUDAERR(cudaMemcpyAsync(U2, &A[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
      CUDAERR(cudaMemcpyAsync(X2, &B[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
      CUDAERR(cudaMemcpyAsync(Y2, &C[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
      cudaEventRecord(evento1, stream1);
      cudaStreamWaitEvent(stream1, evento2);
    } else {
      cudaStreamWaitEvent(stream2, evento1);
      ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[i-1]);
      cudaEventRecord(evento2, stream2);

      CUDAERR(cudaMemcpyAsync(U1, &A[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
      CUDAERR(cudaMemcpyAsync(X1, &B[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
      CUDAERR(cudaMemcpyAsync(Y1, &C[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
      cudaEventRecord(evento1, stream1);
      cudaStreamWaitEvent(stream1, evento2);
    }
  }
  cudaStreamWaitEvent(stream2, evento1);
  if ((NInverses % 2) == 0)
    ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[NInverses-1]);
  else
    ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[NInverses-1]);

  CUDAERR(cudaStreamSynchronize(stream1));
  CUDAERR(cudaStreamSynchronize(stream2));

  CUDAERR(cudaMemcpy(Result, R, sizeof(MyComplex)*NInverses, cudaMemcpyDeviceToHost));

  CUSOLVERERR(cusolverDnDestroy(handle));
  CUDAERR(cudaStreamDestroy(stream1));
  CUDAERR(cudaStreamDestroy(stream2));
  CUDAERR(cudaEventDestroy(evento1));
  CUDAERR(cudaEventDestroy(evento2));

  CUDAERR(cudaFree(TRF)); CUDAERR(cudaFree(Err)); CUDAERR(cudaFree(Y1)); CUDAERR(cudaFree(U1));
  CUDAERR(cudaFree(X1));  CUDAERR(cudaFree(Y2));  CUDAERR(cudaFree(U2)); CUDAERR(cudaFree(X2));
  CUDAERR(cudaFree(R));

  return 0;
}


int RealEvents(const int NChannels, const int NInverses, const MyType *A, const MyType *B, const MyType *C, MyType *Result)
{
  size_t  Size;
  int     *Err=NULL, SizeTRF, i;
  MyType  *U1=NULL, *X1=NULL, *Y1=NULL, *U2=NULL, *X2=NULL, *Y2=NULL, *R=NULL, *TRF=NULL;

  cusolverDnHandle_t handle;
  cudaStream_t       stream1, stream2;
  cudaEvent_t        evento1, evento2;

  CUDAERR(cudaStreamCreate(&stream1));
  CUDAERR(cudaStreamCreate(&stream2));
  CUDAERR(cudaEventCreate(&evento1));
  CUDAERR(cudaEventCreate(&evento2));
  CUSOLVERERR(cusolverDnCreate(&handle));
  CUSOLVERERR(cusolverDnSetStream(handle, stream2));

  Size=(size_t)NChannels * (size_t)NChannels;

  CUDAERR(cudaMalloc((void **)&U1, sizeof(MyType)*Size));
  CUDAERR(cudaMalloc((void **)&X1, sizeof(MyType)*Size));
  CUDAERR(cudaMalloc((void **)&Y1, sizeof(MyType)*Size));
  CUDAERR(cudaMalloc((void **)&U2, sizeof(MyType)*Size));
  CUDAERR(cudaMalloc((void **)&X2, sizeof(MyType)*Size));
  CUDAERR(cudaMalloc((void **)&Y2, sizeof(MyType)*Size));
  CUDAERR(cudaMalloc((void **)&R,  sizeof(MyType)*NInverses));

  CUDAERR(cudaMemcpy(U1, A, sizeof(MyType)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(X1, B, sizeof(MyType)*Size, cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Y1, C, sizeof(MyType)*Size, cudaMemcpyHostToDevice));
  #ifdef SIMPLE
    CUSOLVERERR(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U1, NChannels, &SizeTRF));
  #else
    CUSOLVERERR(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U1, NChannels, &SizeTRF));
  #endif
  CUDAERR(cudaMalloc((void **)&TRF, sizeof(MyType)*SizeTRF));
  CUDAERR(cudaMalloc((void **)&Err, sizeof(int)));

  cudaEventRecord(evento1, stream1);
  for (i=1; i<NInverses; i++)
  {
    if ((i % 2) != 0)
    {
      cudaStreamWaitEvent(stream2, evento1);
      RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[i-1]);
      cudaEventRecord(evento2, stream2);

      CUDAERR(cudaMemcpyAsync(U2, &A[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
      CUDAERR(cudaMemcpyAsync(X2, &B[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
      CUDAERR(cudaMemcpyAsync(Y2, &C[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
      cudaEventRecord(evento1, stream1);
      cudaStreamWaitEvent(stream1, evento2);
    } else {
      cudaStreamWaitEvent(stream2, evento1);
      RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[i-1]);
      cudaEventRecord(evento2, stream2);

      CUDAERR(cudaMemcpyAsync(U1, &A[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
      CUDAERR(cudaMemcpyAsync(X1, &B[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
      CUDAERR(cudaMemcpyAsync(Y1, &C[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
      cudaEventRecord(evento1, stream1);
      cudaStreamWaitEvent(stream1, evento2);
    }
  }
  cudaStreamWaitEvent(stream2, evento1);
  if ((NInverses % 2) == 0)
    RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[NInverses-1]);
  else
    RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[NInverses-1]);

  CUDAERR(cudaStreamSynchronize(stream1));
  CUDAERR(cudaStreamSynchronize(stream2));

  CUDAERR(cudaMemcpy(Result, R, sizeof(MyType)*NInverses, cudaMemcpyDeviceToHost));

  CUSOLVERERR(cusolverDnDestroy(handle));
  CUDAERR(cudaStreamDestroy(stream1));
  CUDAERR(cudaStreamDestroy(stream2));
  CUDAERR(cudaEventDestroy(evento1));
  CUDAERR(cudaEventDestroy(evento2));

  CUDAERR(cudaFree(TRF)); CUDAERR(cudaFree(Err)); CUDAERR(cudaFree(Y1)); CUDAERR(cudaFree(U1));
  CUDAERR(cudaFree(X1));  CUDAERR(cudaFree(Y2));  CUDAERR(cudaFree(U2)); CUDAERR(cudaFree(X2));
  CUDAERR(cudaFree(R));

  return 0;
}


int ComplexEventsOMP(const int NChannels, const int NInverses, const MyComplex *A, const MyComplex *B, const MyComplex *C, MyComplex *Result,  const int Cores)
{
  MyComplex *R=NULL;

  CUDAERR(cudaMalloc((void **)&R, sizeof(MyComplex)*NInverses));

  #ifdef OMP
    #pragma omp parallel num_threads(Cores)
  {
  #endif
    size_t    Size;
    int       *Err=NULL, SizeTRF, i, Chunk, MyID=0, NumTh=1, Start, End;
    MyComplex *U1=NULL, *X1=NULL, *Y1=NULL, *U2=NULL, *X2=NULL, *Y2=NULL, *TRF=NULL;

    cusolverDnHandle_t handle;
    cudaStream_t       stream1, stream2;
    cudaEvent_t        evento1, evento2;

    Size=(size_t)NChannels * (size_t)NChannels;

    CUDAERROMP(cudaStreamCreate(&stream1));
    CUDAERROMP(cudaStreamCreate(&stream2));
    CUDAERROMP(cudaEventCreate(&evento1));
    CUDAERROMP(cudaEventCreate(&evento2));
    CUSOLVERERROMP(cusolverDnCreate(&handle));
    CUSOLVERERROMP(cusolverDnSetStream(handle, stream2));

    #ifdef OMP
      MyID  = omp_get_thread_num();
      NumTh = omp_get_num_threads();
    #endif
    Chunk = NInverses / NumTh;
    Start = MyID * Chunk;

    if (MyID==(NumTh-1)) { End=Start + Chunk + (NInverses % NumTh); } else { End=Start + Chunk; }

    CUDAERROMP(cudaMalloc((void **)&U1, sizeof(MyComplex)*Size));
    CUDAERROMP(cudaMalloc((void **)&X1, sizeof(MyComplex)*Size));
    CUDAERROMP(cudaMalloc((void **)&Y1, sizeof(MyComplex)*Size));
    CUDAERROMP(cudaMalloc((void **)&U2, sizeof(MyComplex)*Size));
    CUDAERROMP(cudaMalloc((void **)&X2, sizeof(MyComplex)*Size));
    CUDAERROMP(cudaMalloc((void **)&Y2, sizeof(MyComplex)*Size));

    CUDAERROMP(cudaMemcpy(U1, &A[Start*Size], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
    CUDAERROMP(cudaMemcpy(X1, &B[Start*Size], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));
    CUDAERROMP(cudaMemcpy(Y1, &C[Start*Size], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice));

    #ifdef SIMPLE
      CUSOLVERERROMP(cusolverDnCpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U1, NChannels, &SizeTRF));
    #else
      CUSOLVERERROMP(cusolverDnZpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U1, NChannels, &SizeTRF));
    #endif
    CUDAERROMP(cudaMalloc((void **)&TRF, sizeof(MyComplex)*SizeTRF));
    CUDAERROMP(cudaMalloc((void **)&Err, sizeof(int)));

    cudaEventRecord(evento1, stream1);
    if ((Start % 2) == 0)
    {
      for(i=(Start+1); i<End; i++)
      {
        if ((i % 2) != 0)
        {
          cudaStreamWaitEvent(stream2, evento1);
          ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[i-1]);
          cudaEventRecord(evento2, stream2);

          CUDAERROMP(cudaMemcpyAsync(U2, &A[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(X2, &B[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(Y2, &C[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          cudaEventRecord(evento1, stream1);
          cudaStreamWaitEvent(stream1, evento2);
        } else {
          cudaStreamWaitEvent(stream2, evento1);
          ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[i-1]);
          cudaEventRecord(evento2, stream2);

          CUDAERROMP(cudaMemcpyAsync(U1, &A[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(X1, &B[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(Y1, &C[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          cudaEventRecord(evento1, stream1);
          cudaStreamWaitEvent(stream1, evento2);
        }
      }
      cudaStreamWaitEvent(stream2, evento1);
      if ((End % 2) == 0)
        ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[End-1]);
      else
        ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[End-1]);
    } else {
      for(i=(Start+1); i<End; i++)
      {
        if ((i % 2) == 0)
        {
          cudaStreamWaitEvent(stream2, evento1);
          ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[i-1]);
          cudaEventRecord(evento2, stream2);

          CUDAERROMP(cudaMemcpyAsync(U2, &A[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(X2, &B[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(Y2, &C[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          cudaEventRecord(evento1, stream1);
          cudaStreamWaitEvent(stream1, evento2);
        } else {
          cudaStreamWaitEvent(stream2, evento1);
          ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[i-1]);
          cudaEventRecord(evento2, stream2);

          CUDAERROMP(cudaMemcpyAsync(U1, &A[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(X1, &B[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(Y1, &C[Size*i], sizeof(MyComplex)*Size, cudaMemcpyHostToDevice, stream1));
          cudaEventRecord(evento1, stream1);
          cudaStreamWaitEvent(stream1, evento2);
        }
      }
      cudaStreamWaitEvent(stream2, evento1);
      if ((End % 2) != 0)
        ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[End-1]);
      else
        ComplexOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[End-1]);
    }

    CUDAERROMP(cudaStreamSynchronize(stream1));
    CUDAERROMP(cudaStreamSynchronize(stream2));

    CUSOLVERERROMP(cusolverDnDestroy(handle));
    CUDAERROMP(cudaStreamDestroy(stream1));
    CUDAERROMP(cudaStreamDestroy(stream2));
    CUDAERROMP(cudaEventDestroy(evento1));
    CUDAERROMP(cudaEventDestroy(evento2));

    CUDAERROMP(cudaFree(TRF)); CUDAERROMP(cudaFree(Err)); CUDAERROMP(cudaFree(Y1)); CUDAERROMP(cudaFree(U1));
    CUDAERROMP(cudaFree(X1));  CUDAERROMP(cudaFree(Y2));  CUDAERROMP(cudaFree(U2)); CUDAERROMP(cudaFree(X2));
  #ifdef OMP
  }
  #endif
  CUDAERR(cudaMemcpy(Result, R, sizeof(MyComplex)*NInverses, cudaMemcpyDeviceToHost));
  CUDAERR(cudaFree(R)); 

  return 0;
}


int RealEventsOMP(const int NChannels, const int NInverses, const MyType *A, const MyType *B, const MyType *C, MyType *Result,  const int Cores)
{
  MyType *R=NULL;

  CUDAERR(cudaMalloc((void **)&R, sizeof(MyType)*NInverses));

  #ifdef OMP
    #pragma omp parallel num_threads(Cores)
  {
  #endif
    size_t  Size;
    int     *Err=NULL, SizeTRF, i, Chunk, MyID=0, NumTh=1, Start, End;
    MyType  *U1=NULL, *X1=NULL, *Y1=NULL, *U2=NULL, *X2=NULL, *Y2=NULL, *TRF=NULL;

    cusolverDnHandle_t handle;
    cudaStream_t       stream1, stream2;
    cudaEvent_t        evento1, evento2;

    Size=(size_t)NChannels * (size_t)NChannels;

    CUDAERROMP(cudaStreamCreate(&stream1));
    CUDAERROMP(cudaStreamCreate(&stream2));
    CUDAERROMP(cudaEventCreate(&evento1));
    CUDAERROMP(cudaEventCreate(&evento2));
    CUSOLVERERROMP(cusolverDnCreate(&handle));
    CUSOLVERERROMP(cusolverDnSetStream(handle, stream2));

    #ifdef OMP
      MyID  = omp_get_thread_num();
      NumTh = omp_get_num_threads();
    #endif
    Chunk = NInverses / NumTh;
    Start = MyID * Chunk;

    if (MyID==(NumTh-1)) { End=Start + Chunk + (NInverses % NumTh); } else { End=Start + Chunk; }

    CUDAERROMP(cudaMalloc((void **)&U1, sizeof(MyType)*Size));
    CUDAERROMP(cudaMalloc((void **)&X1, sizeof(MyType)*Size));
    CUDAERROMP(cudaMalloc((void **)&Y1, sizeof(MyType)*Size));
    CUDAERROMP(cudaMalloc((void **)&U2, sizeof(MyType)*Size));
    CUDAERROMP(cudaMalloc((void **)&X2, sizeof(MyType)*Size));
    CUDAERROMP(cudaMalloc((void **)&Y2, sizeof(MyType)*Size));

    CUDAERROMP(cudaMemcpy(U1, &A[Start*Size], sizeof(MyType)*Size, cudaMemcpyHostToDevice));
    CUDAERROMP(cudaMemcpy(X1, &B[Start*Size], sizeof(MyType)*Size, cudaMemcpyHostToDevice));
    CUDAERROMP(cudaMemcpy(Y1, &C[Start*Size], sizeof(MyType)*Size, cudaMemcpyHostToDevice));

    #ifdef SIMPLE
      CUSOLVERERROMP(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U1, NChannels, &SizeTRF));
    #else
      CUSOLVERERROMP(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, NChannels, U1, NChannels, &SizeTRF));
    #endif
    CUDAERROMP(cudaMalloc((void **)&TRF, sizeof(MyType)*SizeTRF));
    CUDAERROMP(cudaMalloc((void **)&Err, sizeof(int)));

    cudaEventRecord(evento1, stream1);
    if ((Start % 2) == 0)
    {
      for(i=(Start+1); i<End; i++)
      {
        if ((i % 2) != 0)
        {
          cudaStreamWaitEvent(stream2, evento1);
          RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[i-1]);
          cudaEventRecord(evento2, stream2);

          CUDAERROMP(cudaMemcpyAsync(U2, &A[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(X2, &B[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(Y2, &C[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          cudaEventRecord(evento1, stream1);
          cudaStreamWaitEvent(stream1, evento2);
        } else {
          cudaStreamWaitEvent(stream2, evento1);
          RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[i-1]);
          cudaEventRecord(evento2, stream2);

          CUDAERROMP(cudaMemcpyAsync(U1, &A[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(X1, &B[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(Y1, &C[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          cudaEventRecord(evento1, stream1);
          cudaStreamWaitEvent(stream1, evento2);
        }
      }
      cudaStreamWaitEvent(stream2, evento1);
      if ((End % 2) == 0)
        RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[End-1]);
      else
        RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[End-1]);
    } else {
      for(i=(Start+1); i<End; i++)
      {
        if ((i % 2) == 0)
        {
          cudaStreamWaitEvent(stream2, evento1);
          RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[i-1]);
          cudaEventRecord(evento2, stream2);

          CUDAERROMP(cudaMemcpyAsync(U2, &A[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(X2, &B[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(Y2, &C[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          cudaEventRecord(evento1, stream1);
          cudaStreamWaitEvent(stream1, evento2);
        } else {
          cudaStreamWaitEvent(stream2, evento1);
          RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[i-1]);
          cudaEventRecord(evento2, stream2);

          CUDAERROMP(cudaMemcpyAsync(U1, &A[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(X1, &B[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          CUDAERROMP(cudaMemcpyAsync(Y1, &C[Size*i], sizeof(MyType)*Size, cudaMemcpyHostToDevice, stream1));
          cudaEventRecord(evento1, stream1);
          cudaStreamWaitEvent(stream1, evento2);
        }
      }
      cudaStreamWaitEvent(stream2, evento1);
      if ((End % 2) != 0)
        RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U2, X2, Y2, TRF, &R[End-1]);
      else
        RealOneStreamStep(handle, stream2, NChannels, SizeTRF, Err, U1, X1, Y1, TRF, &R[End-1]);
    }

    CUDAERROMP(cudaStreamSynchronize(stream1));
    CUDAERROMP(cudaStreamSynchronize(stream2));

    CUSOLVERERROMP(cusolverDnDestroy(handle));
    CUDAERROMP(cudaStreamDestroy(stream1));
    CUDAERROMP(cudaStreamDestroy(stream2));
    CUDAERROMP(cudaEventDestroy(evento1));
    CUDAERROMP(cudaEventDestroy(evento2));

    CUDAERROMP(cudaFree(TRF)); CUDAERROMP(cudaFree(Err)); CUDAERROMP(cudaFree(Y1)); CUDAERROMP(cudaFree(U1));
    CUDAERROMP(cudaFree(X1));  CUDAERROMP(cudaFree(Y2));  CUDAERROMP(cudaFree(U2)); CUDAERROMP(cudaFree(X2));
  #ifdef OMP
  }
  #endif
  CUDAERR(cudaMemcpy(Result, R, sizeof(MyType)*NInverses, cudaMemcpyDeviceToHost));
  CUDAERR(cudaFree(R)); 

  return 0;
}
