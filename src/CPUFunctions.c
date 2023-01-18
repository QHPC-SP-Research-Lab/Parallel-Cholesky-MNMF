#include "CPUdefines.h"


double dseconds()
{
  struct timeval tm;

  gettimeofday(&tm, NULL);

  return tm.tv_sec + tm.tv_usec/1.0E6;
}


/* In future, do better */
inline MyComplex cmul(MyComplex z, MyComplex w)
{
  MyComplex x;

  #ifdef MKL
    x.real=z.real*w.real - z.imag*w.imag;
    x.imag=z.real*w.imag + z.imag*w.real;
  #else
    x = (creal(z) * creal(w) - cimag(z) * cimag(w)) + (creal(z) * cimag(w) + cimag(z) * creal(w))*I;
  #endif    
  return x;
}


/* In future, do better */
inline MyComplex cdiv(MyComplex z, MyComplex w)
{
  MyComplex x;
  MyType    y;

  #ifdef MKL
    y=w.real*w.real + w.imag*w.imag;

    x.real=(z.real*w.real + z.imag*w.imag) / y;
    x.imag=(z.imag*w.real - z.real*w.imag) / y;
  #else
    y=creal(w) * creal(w) + cimag(w) * cimag(w);

    x=((creal(z) * creal(w) + cimag(z) * cimag(w)) / y) + ((cimag(z) * creal(w) - creal(z) * cimag(w)) / y)*I;
  #endif  
  return x;
}


void ComplexCPU(const int NChannels, const int NInverses, const MyComplex *A, const MyComplex *B, const MyComplex *C, MyComplex *Res, const int OmpCores, const int LibCores)
{
  /* The API functions used by MATLAB Memory Manager (mxCreateDoubleMatrix, etc.) are not thread-safe!.
     That is, you should never use any API functions that allocate memory inside of a parallel thread */
  #ifdef MKL
    #ifndef OMP
      mkl_set_num_threads(OmpCores * LibCores);
    #else
      mkl_set_num_threads(LibCores);
    #endif  
  #else
    #ifndef OMP
      openblas_set_num_threads(OmpCores * LibCores);
    #else
      openblas_set_num_threads(LibCores);
    #endif  
  #endif

  #ifdef OMP
    #pragma omp parallel num_threads(OmpCores)
  {
  #endif
    int i, j, Chunk, MyID=0, NumTh=1, Start, End;
    size_t Size;

    MyComplex *U=NULL, *X=NULL, *Y=NULL, Den;

    #ifdef MKL
      MyComplex Uno;
      Uno.real = 1.0; Uno.imag = 0.0;
    #else
      MyType Uno[2]={1.0, 0.0};
    #endif

    Size=(size_t)NChannels * (size_t)NChannels;

    #ifdef OMP
      MyID  = omp_get_thread_num();
      NumTh = omp_get_num_threads();
    #endif
    Chunk = NInverses / NumTh;
    Start = MyID * Chunk;

    if (MyID==(NumTh-1)) { End=Start + Chunk + (NInverses % NumTh); } else { End=Start + Chunk; }

    U =(MyComplex *)malloc(Size*sizeof(MyComplex));  
    if (U == NULL) mexErrMsgIdAndTxt("MATLAB:ComplexCPU:NULL", "Out off memory.");

    X =(MyComplex *)malloc(Size*sizeof(MyComplex));  
    if (X == NULL) mexErrMsgIdAndTxt("MATLAB:ComplexCPU:NULL", "Out off memory.");

    Y =(MyComplex *)malloc(Size*sizeof(MyComplex));  
    if (Y == NULL) mexErrMsgIdAndTxt("MATLAB:ComplexCPU:NULL", "Out off memory.");

    for(i=Start; i<End; i++)
    {
      memcpy(U, &A[Size*i], sizeof(MyComplex)*Size);
      memcpy(X, &B[Size*i], sizeof(MyComplex)*Size);
      memcpy(Y, &C[Size*i], sizeof(MyComplex)*Size);

      /* Compute inv(A)*B*inv(A)*C. A is symmetric positive definite
         1) Cholesky Factorization. Input: A (stored in U). Output: U */
      #ifdef SIMPLE
        LAPACKE_cpotrf(LAPACK_COL_MAJOR, 'U', NChannels, U, NChannels);
      #else
        LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U', NChannels, U, NChannels);
      #endif
 
      /* 2) Solve the system AX=B via one call to POTRS. B stored in X. Result in X */
      #ifdef SIMPLE
        LAPACKE_cpotrs(LAPACK_COL_MAJOR, 'U', NChannels, NChannels , U, NChannels, X, NChannels);
      #else
        LAPACKE_zpotrs(LAPACK_COL_MAJOR, 'U', NChannels, NChannels , U, NChannels, X, NChannels);
      #endif

      /* 3) Solve the system AX=C via one call to POTRS. C stored in Y. Result in Y */
      #ifdef SIMPLE
        LAPACKE_cpotrs(LAPACK_COL_MAJOR, 'U', NChannels, NChannels , U, NChannels, Y, NChannels);
      #else
        LAPACKE_zpotrs(LAPACK_COL_MAJOR, 'U', NChannels, NChannels , U, NChannels, Y, NChannels);
      #endif

      /* In-place. It saves little time, but it is used by analogy to the GPU */
      #ifdef SIMPLE
        #ifdef MKL
          mkl_cimatcopy('C', 'T', NChannels, NChannels, Uno, X, NChannels, NChannels);
        #else
          cblas_cimatcopy(CblasColMajor, CblasTrans, NChannels, NChannels, Uno, (double *)X, NChannels, NChannels);
        #endif
        cblas_cdotu_sub(NChannels*NChannels, X, 1, Y, 1, &Res[i]);
      #else
        #ifdef MKL
          mkl_zimatcopy('C', 'T', NChannels, NChannels, Uno, X, NChannels, NChannels);
        #else
          cblas_zimatcopy(CblasColMajor, CblasTrans, NChannels, NChannels, Uno, (double *)X, NChannels, NChannels);
        #endif
        cblas_zdotu_sub(NChannels*NChannels, X, 1, Y, 1, &Res[i]);
      #endif

      #ifdef MKL
        Den.real=0; Den.imag=0;
        for(j=0;j<NChannels;j++) { Den.real += Y[j*NChannels+j].real; Den.imag += Y[j*NChannels+j].imag; }
      #else
        Den = 0.0 + 0.0*I;
        for(j=0;j<NChannels;j++) { Den += creal(Y[j*NChannels+j]) + cimag(Y[j*NChannels+j])*I; }
      #endif
      Res[i]=cdiv(Res[i], Den);
    }
    free(U); free(X); free(Y);
  #ifdef OMP
  }
  #endif
}


void RealCPU(const int NChannels, const int NInverses, const MyType *A, const MyType *B, const MyType *C, MyType *Res, const int OmpCores, const int LibCores)
{
  /* The API functions used by MATLAB Memory Manager (mxCreateDoubleMatrix, etc.) are not thread-safe!.
     That is, you should never use any API functions that allocate memory inside of a parallel thread */
  #ifdef MKL
    #ifndef OMP
      mkl_set_num_threads(OmpCores * LibCores);
    #else
      mkl_set_num_threads(LibCores);
    #endif  
  #else
    #ifndef OMP
      openblas_set_num_threads(OmpCores * LibCores);
    #else
      openblas_set_num_threads(LibCores);
    #endif  
  #endif

  #ifdef OMP
    #pragma omp parallel num_threads(OmpCores)
  {
  #endif
    int i, j, Chunk, MyID=0, NumTh=1, Start, End;
    size_t Size;

    MyType *U=NULL, *X=NULL, *Y=NULL, Den;
    
    Size  = (size_t)NChannels * (size_t)NChannels;

    #ifdef OMP
      MyID  = omp_get_thread_num();
      NumTh = omp_get_num_threads();
    #endif
    Chunk = NInverses / NumTh;
    Start = MyID * Chunk;

    if (MyID==(NumTh-1)) { End=Start + Chunk + (NInverses % NumTh); } else { End=Start + Chunk; }

    U =(MyType *)malloc(Size*sizeof(MyType));  
    if (U == NULL) mexErrMsgIdAndTxt("MATLAB:RealCPU:NULL", "Out off memory.");

    X =(MyType *)malloc(Size*sizeof(MyType));  
    if (X == NULL) mexErrMsgIdAndTxt("MATLAB:RealCPU:NULL", "Out off memory.");

    Y =(MyType *)malloc(Size*sizeof(MyType));  
    if (Y == NULL) mexErrMsgIdAndTxt("MATLAB:RealCPU:NULL", "Out off memory.");

    for(i=Start; i<End; i++)
    {
      memcpy(U, &A[Size*i], sizeof(MyType)*Size);
      memcpy(X, &B[Size*i], sizeof(MyType)*Size);
      memcpy(Y, &C[Size*i], sizeof(MyType)*Size);

      /* Compute inv(A)*B*inv(A)*C. A is symmetric positive definite
         1) Cholesky Factorization. Input: A (stored in U). Output: U */
      #ifdef SIMPLE
        LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', NChannels, U, NChannels);
      #else
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', NChannels, U, NChannels);
      #endif
  
      /* 2) Solve the system AX=B via one call to POTRS. B stored in X. Result in X */
      #ifdef SIMPLE
        LAPACKE_spotrs(LAPACK_COL_MAJOR, 'U', NChannels, NChannels, U, NChannels, X, NChannels);
      #else
        LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', NChannels, NChannels, U, NChannels, X, NChannels);
      #endif

      /* 3) Solve the system AX=C via one call to POTRS. C stored in Y. Result in Y */
      #ifdef SIMPLE
        LAPACKE_spotrs(LAPACK_COL_MAJOR, 'U', NChannels, NChannels, U, NChannels, Y, NChannels);
      #else
        LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', NChannels, NChannels, U, NChannels, Y, NChannels);
      #endif

      /* In-place. It saves little time, but it is used by analogy to the GPU */
      #ifdef SIMPLE
        #ifdef MKL
          mkl_simatcopy('C', 'T', NChannels, NChannels, 1.0, X, NChannels, NChannels);
        #else
          cblas_simatcopy(CblasColMajor, CblasTrans, NChannels, NChannels, 1.0, X, NChannels, NChannels);
        #endif
        Res[i]=cblas_sdot(NChannels*NChannels, X, 1, Y, 1);
      #else
        #ifdef MKL
          mkl_dimatcopy('C', 'T', NChannels, NChannels, 1.0, X, NChannels, NChannels);
        #else
          cblas_dimatcopy(CblasColMajor, CblasTrans, NChannels, NChannels, 1.0, X, NChannels, NChannels);
        #endif
        Res[i]=cblas_ddot(NChannels*NChannels, X, 1, Y, 1);
      #endif
      Den=0.0;
      for(j=0;j<NChannels;j++) { Den += Y[j*NChannels+j]; }
      Res[i]=Res[i] / Den;
    }
    free(U); free(X); free(Y);
  #ifdef OMP
  }
  #endif
}
