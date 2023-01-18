#!/bin/bash
export OMP_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=2
export OPENBLAS_MAIN_FREE=1

if [ ! -s Multinversa.m ]; then
   echo "Error, Multinversa.m does no exist"
   exit 1
fi

cd src
rm -f CPUSolver.mexa64
make -f MakefileOpenBlas

if [ ! -x CPUSolver.mexa64 ]; then
   echo "Error, CPUSolver.mexa64 does no exist"
   exit 1
fi

cd ..

TMP=/opt/openblasNoOMP-0.3.21/lib
OpenBlas="$TMP/libopenblas.so"

#
# Arch = 1 --> CPU, Other value --> GPU
#
# Model: Only used with GPU, for now
#       | Model = 1 --> Streams
#       | Model = 2 --> ZeroCopy
#   GPU | Model = 3 --> Classic
#       | Model = 4 --> With OpenMP
#       | Model = 5 --> With Events
#       | Model = 6 --> With Events and OpenMP
#
# DataType = 1 --> Real numbers (double), Other value --> Complex
#
Arch="1"
Model="1"
Tipo="2"
CHANS=(   100  500 1000 2000 3000 4000 6000)
 NINV=(224000 8960 2208  512  192   96   32)
Sizes=7

PROCS=(32 16 8 4  2  1)
MKLCS=( 1  2 4 8 16 32)
CPUss=6

pos=0
while [ $pos -lt $CPUss ]
do
  Cores=${PROCS[pos]}
  MKLcr=${MKLCS[$pos]}

  export OMP_NUM_THREADS=$Cores
  export OPENBLAS_NUM_THREADS=$MKLcr

  for Tipo in $Tipo
  do
    actual=0
    while [ $actual -lt $Sizes ]
    do
      nchan=${CHANS[actual]} 
      ninve=${NINV[actual]}

      echo "NChannels: $nchan, NInverses: $ninve, Architecture: $Arch, Model: $Model, Data_Type: $Tipo, OmpCores: $Cores MklCores: $MKLcr"
      LD_PRELOAD="$OpenBlas" matlab -nojvm -nodisplay -r "Multinversa($nchan, $ninve, $Arch, $Model, $Tipo, $Cores, $MKLcr);exit;"

      actual=$(( $actual + 1 ))
    done
  done
  pos=$(( $pos + 1 ))
done


PROCS=(16 8 4 2  1)
MKLCS=( 1 2 4 8 16)
CPUss=5

pos=0
while [ $pos -lt $CPUss ]
do
  Cores=${PROCS[pos]}
  MKLcr=${MKLCS[$pos]}

  export OMP_NUM_THREADS=$Cores
  export OPENBLAS_NUM_THREADS=$MKLcr

  for Tipo in $Tipo
  do
    actual=0
    while [ $actual -lt $Sizes ]
    do
      nchan=${CHANS[actual]} 
      ninve=${NINV[actual]}

      echo "NChannels: $nchan, NInverses: $ninve, Architecture: $Arch, Model: $Model, Data_Type: $Tipo, OmpCores: $Cores MklCores: $MKLcr"
      LD_PRELOAD="$OpenBlas" matlab -nojvm -nodisplay -r "Multinversa($nchan, $ninve, $Arch, $Model, $Tipo, $Cores, $MKLcr);exit;"

      actual=$(( $actual + 1 ))
    done
  done
  pos=$(( $pos + 1 ))
done


PROCS=(8 4 2 1)
MKLCS=(1 2 4 8)
CPUss=4

pos=0
while [ $pos -lt $CPUss ]
do
  Cores=${PROCS[pos]}
  MKLcr=${MKLCS[$pos]}

  export OMP_NUM_THREADS=$Cores
  export OPENBLAS_NUM_THREADS=$MKLcr

  for Tipo in $Tipo
  do
    actual=0
    while [ $actual -lt $Sizes ]
    do
      nchan=${CHANS[actual]} 
      ninve=${NINV[actual]}

      echo "NChannels: $nchan, NInverses: $ninve, Architecture: $Arch, Model: $Model, Data_Type: $Tipo, OmpCores: $Cores MklCores: $MKLcr"
      LD_PRELOAD="$OpenBlas" matlab -nojvm -nodisplay -r "Multinversa($nchan, $ninve, $Arch, $Model, $Tipo, $Cores, $MKLcr);exit;"

      actual=$(( $actual + 1 ))
    done
  done
  pos=$(( $pos + 1 ))
done


PROCS=(4 2 1)
MKLCS=(1 2 4)
CPUss=3

pos=0
while [ $pos -lt $CPUss ]
do
  Cores=${PROCS[pos]}
  MKLcr=${MKLCS[$pos]}

  export OMP_NUM_THREADS=$Cores
  export OPENBLAS_NUM_THREADS=$MKLcr

  for Tipo in $Tipo
  do
    actual=0
    while [ $actual -lt $Sizes ]
    do
      nchan=${CHANS[actual]} 
      ninve=${NINV[actual]}

      echo "NChannels: $nchan, NInverses: $ninve, Architecture: $Arch, Model: $Model, Data_Type: $Tipo, OmpCores: $Cores MklCores: $MKLcr"
      LD_PRELOAD="$OpenBlas" matlab -nojvm -nodisplay -r "Multinversa($nchan, $ninve, $Arch, $Model, $Tipo, $Cores, $MKLcr);exit;"

      actual=$(( $actual + 1 ))
    done
  done
  pos=$(( $pos + 1 ))
done

PROCS=(2 1)
MKLCS=(1 2)
CPUss=2

pos=0
while [ $pos -lt $CPUss ]
do
  Cores=${PROCS[pos]}
  MKLcr=${MKLCS[$pos]}

  export OMP_NUM_THREADS=$Cores
  export OPENBLAS_NUM_THREADS=$MKLcr

  for Tipo in $Tipo
  do
    actual=0
    while [ $actual -lt $Sizes ]
    do
      nchan=${CHANS[actual]} 
      ninve=${NINV[actual]}

      echo "NChannels: $nchan, NInverses: $ninve, Architecture: $Arch, Model: $Model, Data_Type: $Tipo, OmpCores: $Cores MklCores: $MKLcr"
      LD_PRELOAD="$OpenBlas" matlab -nojvm -nodisplay -r "Multinversa($nchan, $ninve, $Arch, $Model, $Tipo, $Cores, $MKLcr);exit;"

      actual=$(( $actual + 1 ))
    done
  done
  pos=$(( $pos + 1 ))
done


PROCS=(1)
MKLCS=(1)
CPUss=1

pos=0
while [ $pos -lt $CPUss ]
do
  Cores=${PROCS[pos]}
  MKLcr=${MKLCS[$pos]}

  export OMP_NUM_THREADS=$Cores
  export OPENBLAS_NUM_THREADS=$MKLcr

  for Tipo in $Tipo
  do
    actual=0
    while [ $actual -lt $Sizes ]
    do
      nchan=${CHANS[actual]} 
      ninve=${NINV[actual]}

      echo "NChannels: $nchan, NInverses: $ninve, Architecture: $Arch, Model: $Model, Data_Type: $Tipo, OmpCores: $Cores MklCores: $MKLcr"
      LD_PRELOAD="$OpenBlas" matlab -nojvm -nodisplay -r "Multinversa($nchan, $ninve, $Arch, $Model, $Tipo, $Cores, $MKLcr);exit;"

      actual=$(( $actual + 1 ))
    done
  done
  pos=$(( $pos + 1 ))
done
