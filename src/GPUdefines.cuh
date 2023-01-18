#pragma once

#ifndef GPU_INV
#define GPU_INV

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>

#include "mex.h"

#ifdef SIMPLE
  #define MyType    float
  #define MyComplex cuFloatComplex
#else
  #define MyType    double
  #define MyComplex cuDoubleComplex
#endif
#define MyFFTGPUType cufftHandle

#ifdef OCTAVE
  #define mxGetDoubles mxGetPr
#endif

#ifndef min
  #define min(x,y) ((x < y) ? x : y)
#endif

#ifndef max
  #define max(x,y) ((x > y) ? x : y)
#endif

#define sizeWarp         32
#define halfWarp         16
#define MASK     0xffffffff

__inline__ __device__ MyType ReduceWarp(MyType val) {
  for (unsigned int offset = halfWarp; offset > 0; offset /= 2)
    val += __shfl_down_sync(MASK, val, offset, sizeWarp);
  return val;
}

#define CUDAERR(x) do { if((x)!=cudaSuccess) { \
   printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(x), __FILE__, __LINE__);\
   return EXIT_FAILURE;}} while(0)

#define CUDAERROMP(x) do { if((x)!=cudaSuccess) { \
   printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(x), __FILE__, __LINE__);\
   }} while(0)

#define CUBLASERR(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
   printf("CUBLAS error: %s, line %d\n", __FILE__, __LINE__);\
   return EXIT_FAILURE;}} while(0)

#define CUSOLVERERR(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) { \
   printf("cuSolver error: %s, line %d\n", __FILE__, __LINE__);\
   return EXIT_FAILURE;}} while(0)

#define CUSOLVERERROMP(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) { \
   printf("cuSolver error: %s, line %d\n", __FILE__, __LINE__);\
   }} while(0)

int    RealOpenMP(const int, const int, const MyType*, const MyType*, const MyType*, MyType*, const int);
int   RealClassic(const int, const int, const MyType*, const MyType*, const MyType*, MyType*);
int  RealZeroCopy(const int, const int,       MyType*,       MyType*,       MyType*, MyType*);
int   RealStreams(const int, const int, const MyType*, const MyType*, const MyType*, MyType*, const int);
int    RealEvents(const int, const int, const MyType*, const MyType*, const MyType*, MyType*);
int RealEventsOMP(const int, const int, const MyType*, const MyType*, const MyType*, MyType*, const int);

int    ComplexOpenMP(const int, const int, const MyComplex*, const MyComplex*, const MyComplex*, MyComplex*, const int);
int   ComplexClassic(const int, const int, const MyComplex*, const MyComplex*, const MyComplex*, MyComplex*);
int  ComplexZeroCopy(const int, const int,       MyComplex*,       MyComplex*,       MyComplex*, MyComplex*);
int   ComplexStreams(const int, const int, const MyComplex*, const MyComplex*, const MyComplex*, MyComplex*, const int);
int    ComplexEvents(const int, const int, const MyComplex*, const MyComplex*, const MyComplex*, MyComplex*);
int ComplexEventsOMP(const int, const int, const MyComplex*, const MyComplex*, const MyComplex*, MyComplex*, const int);

#endif
