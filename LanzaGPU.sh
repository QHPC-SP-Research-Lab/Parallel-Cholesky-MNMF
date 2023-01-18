#!/bin/bash

if [ ! -s Multinversa.m ]; then
   echo "Error, Multinversa.m does no exist"
   exit 1
fi

cd src
rm -f GPUSolver.mexa64
make -f MakefileGPU

if [ ! -x GPUSolver.mexa64 ]; then
   echo "Error, GPUSolver.mexa64 does no exist"
   exit 1
fi

cd ..

GPU="/opt/cuda-11.4/lib64/libcublas.so /opt/cuda-11.4/targets/x86_64-linux/lib/libcublasLt.so.11"

#
# Arch = 1 --> CPU, Other value --> GPU
#
# Model: Only used with GPU, for now
#       | Model = 1 --> Streams
#       | Model = 2 --> ZeroCopy (very bad)
#       | Model = 3 --> Classic
#   GPU | Model = 4 --> With OMP
#       | Model = 5 --> With Events
#       | Model = 6 --> With Events and OMP
#
# DataType = 1 --> Real numbers (double), Other value --> Complex
#
Arch="2"
Type="2"
Streams="24"
Model="1 4 5 3"
Ncores=32

CHANS=(    64    128   256  512 1024 2048 4096)
 NINV=(524288 131072 32768 8192 2048  480   96)
Sizes=7

for arq in $Arch
do
  for tip in $Type
  do
    for mod in $Model
    do
      pos=0
      while [ $pos -lt $Sizes ]
      do
        ncha=${CHANS[pos]} 
        ninv=${NINV[pos]}

        echo "NChannels: $ncha, NInverses: $ninv, Architecture: $arq, Data: $tip, Model: $mod, Cores: $Ncores, Streams: $Streams"
        LD_PRELOAD="$GPU" matlab -nojvm -nodisplay -r "Multinversa($ncha, $ninv, $arq, $mod, $tip, $Ncores, $Streams);exit;"

        pos=$(( $pos + 1 ))
      done
    done
  done
done
