%
% Arch = 1 --> CPU, Other value --> GPU
%
% Model: Only used with GPU, for now
%       | Model = 1 --> Streams
%       | Model = 2 --> ZeroCopy
%   GPU | Model = 3 --> Classic
%       | Model = 4 --> With OpenMP
%       | Model = 5 --> With Events
%       | Model = 6 --> With Events and OpenMP
%
% DataType = 1 --> Real numbers (double), Other value --> Complex
%
%
% OmpCores = number of threads in OpenMP parallel constructors (for CPU and % GPU)
% LibCores = number of
%              CPU --> cores used within API functions (that is, within MKL/Openblas API functions)
%              GPU --> CPU cores in Model 4 and 6; streams in Model 1; not used in other models
%
function Multinversa(NChannels, NInverses, Arch, Model, DataType, OmpCores, LibCores)
  IsOctave=exist('OCTAVE_VERSION', 'builtin') ~= 0;
  if(IsOctave)
     fprintf('\nDo not work with OCTAVE\n');
     exit;
  end

  if(Arch==1)
     maxNumCompThreads(OmpCores*LibCores);
  else
     maxNumCompThreads(OmpCores);
  end
  fprintf('Using %d channels, %d inverses and %d cores.\n', NChannels, NInverses, maxNumCompThreads);

  tic
     if(DataType==1)
        fprintf('Start building random real matrices and (after) real symetric matrices. ');
     else
        fprintf('Start building random complex matrices and (after) hermitian matrices. ');
     end
     TMAT = zeros(1, NInverses);
     DIAG = diag(ones(1, NChannels));

     if(DataType==1)
        A = rand(NChannels, NChannels);
        B = rand(NChannels*NChannels*NInverses, 1);
        C = rand(NChannels*NChannels*NInverses, 1);
        A = pagemtimes(A,'none',A,'transpose');
     else
        A = complex(rand(NChannels, NChannels), rand(NChannels, NChannels));
        B = complex(rand(NChannels*NChannels*NInverses, 1), rand(NChannels*NChannels*NInverses, 1));
        C = complex(rand(NChannels*NChannels*NInverses, 1), rand(NChannels*NChannels*NInverses, 1));
        A = pagemtimes(A,'none',A,'ctranspose');
     end
     A = A/max(A(:));
     A = A+2*DIAG;
     A = repmat(A, [1 1 NInverses]);

     B = reshape(B, [NChannels, NChannels, NInverses]);
     C = reshape(C, [NChannels, NChannels, NInverses]);
  time=toc;
  quetenemos = whos;
  memoria=sum([quetenemos.bytes]) / (1024.0*1204.0*1024.0);
  fprintf('Finish in %1.5f sec. GBytes used: %1.5f GB\n', time, memoria);

  fprintf('Start MATLAB solver. ');
  tic
     for i=1:NInverses
        U = chol(A(:,:,i));
        X = U\(U'\B(:,:,i));
        Y = U\(U'\C(:,:,i));
        TMAT(i) = trace(X*Y)/trace(Y);
     end
  time=toc;
  quetenemos = whos;
  memoria=sum([quetenemos.bytes]) / (1024.0*1204.0*1024.0);
  fprintf('Finish in %1.5f sec (%1.5f sec per inverse). GBytes used: %1.5f GB\n', time, time/NInverses, memoria);

  addpath('./src');

  % Arch=1 --> CPU, otherwise --> GPU
  if(Arch==1)  
     fprintf('Start CPU solver using data type %1d, Omp cores %2d and Lib cores %2d. ', DataType, OmpCores, LibCores);
     tic
       TCG=CPUSolver(A, B, C, DataType, OmpCores, LibCores);
     time=toc;
     quetenemos = whos;
     memoria=sum([quetenemos.bytes]) / (1024.0*1204.0*1024.0);
     fprintf('Finish in %1.5f sec (%1.5f ser per inverse). GBytes used: %1.5f GB.\n', time, time/NInverses, memoria);
  else
     fprintf('Start GPU solver using model %1d and data type %1d. ', Model, DataType);
     tic
       TCG=GPUSolver(A, B, C, Model, DataType, OmpCores, LibCores);
     time=toc;
     quetenemos = whos;
     memoria=sum([quetenemos.bytes]) / (1024.0*1204.0*1024.0);
     fprintf('Finish in %1.5f sec (%1.5f sec per inverse). GBytes used: %1.5f GB.\n', time, time/NInverses, memoria);
  end
  fprintf('Error Matlab vs Custom: %1.15f\n', norm(TMAT-TCG, 'fro'));
end
