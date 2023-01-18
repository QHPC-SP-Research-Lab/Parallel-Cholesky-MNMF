#pragma once

#ifndef CPU_INV
#define CPU_INV

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <complex.h>
#include <omp.h>
#include "mex.h"

#ifdef OCTAVE
  #define mxGetDoubles mxGetPr
#endif

#ifdef MKL
  #include <mkl.h>
  #ifdef SIMPLE
    #define MyType    float
    #define MyComplex MKL_Complex8
  #else
    #define MyType    double
    #define MyComplex MKL_Complex16
  #endif
#else
  #include <cblas.h>
  #include <lapacke.h>
  #ifdef SIMPLE
    #define MyType    float
    #define MyComplex float complex
  #else
    #define MyType    double
    #define MyComplex double complex
  #endif
#endif

#ifndef min
  #define min(x,y) ((x < y) ? x : y)
#endif

#ifndef max
  #define max(x,y) ((x > y) ? x : y)
#endif

double dseconds();

inline MyComplex cmul(MyComplex, MyComplex);
inline MyComplex cdiv(MyComplex, MyComplex);

void    RealCPU(const int, const int, const MyType*,    const MyType*,    const MyType*,    MyType*,    const int, const int);
void ComplexCPU(const int, const int, const MyComplex*, const MyComplex*, const MyComplex*, MyComplex*, const int, const int);

#endif
