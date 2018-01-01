%% Loading data

addpath('../../Tests');
addpath('../../ThirdParty/SeismicLab/codes/radon_transforms/')
addpath('../../ThirdParty/SegyMAT/')
addpath('../../../IA353/NeuralNetwork/')
addpath('../../../IA353/ExtremeLearningMachine/')
addpath('../../../IA353/EchoStateNetworks/')
addpath('../../../IA353/Regularization/')

load('CaseData1_0/tracos_in_radon');
load('CaseData1_0/parameter');
load('CaseData1_0/full_radon_trace_zero_offset_CaseData1_0.mat')

%% Case two primary and multiples - Zero offset

% Ploting filtered trace and reference trace
traces_matrix = radon_p1p2_sec_mul_div;
traces_matrix_prim = radon_p1p2_primaries_div;

figure(1)
imagesc(traces_matrix, [-1 1]*1e-5)
title('Primaries and multiples')
axis([0 500 0 1000])
grid

figure(2)
imagesc(traces_matrix_prim, [-1 1]*1e-5)
title('Primaries')
axis([0 500 0 1000])
grid

%% Filtering all traces

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;
traces_matrix_prim = zeros(size(traces_matrix));

% Using cursor information
prediction_step = 0;
offset = 0;
filter_len = 10;   
mid_layer_sz = 45;
regularization = 0;
initial_weigths_amp = 0.5;

deconvolved_matrix = zeros(size(traces_matrix));
trace_nb = size(traces_matrix, 2);
mse_prediction = zeros(trace_nb, 1);
mse_reference_trace = zeros(trace_nb, 1);

for i=1:trace_nb
  
  % Nomalizing data
  [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, i, samples_start, attenuation_factor);
  trace_norm_prim = trace_pre_processing(traces_matrix_prim, i, samples_start, attenuation_factor);

  % Neural network setup
  clear nn
  in_sz = filter_len;
  out_sz = 1;
  nn.func = @tanh;
  nn.b = 0;

  nn.v = initial_weigths_amp*(rand(in_sz+1, mid_layer_sz));
  nn = neuro_net_init(nn);

  % Preparing data based in parameters
  % Using info from cursor
  prediction_step = predic_step(i)-offset;
  [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);

  % Calculating extreme learning machines values
  nn.w = calc_elm_weigths(train_set, target, regularization, nn)';

  % Apply network to all traces
  deconvolved_matrix(:, i) = target - neural_nete(train_set, nn);
  mse_prediction(i) = mean(deconvolved_matrix(:, i).^2);
  mse_reference_trace(i) = mean((deconvolved_matrix(:, i) - trace_norm_prim).^2);
  deconvolved_matrix(:, i) = deconvolved_matrix(:, i)*max_amp*std_dev + avg;

end

%% Show reference trace and filtered traces in Radon domain

figure(3)
imagesc(traces_matrix, [-1 1]*1e-5)
title('Primaries')
axis([0 500 0 1000])
grid

figure(4)
imagesc(deconvolved_matrix, [-1 1]*1e-5)
title('Filtered trace')
axis([0 500 0 1000])
grid

%% Inverting traces with only primaries to use as reference

primaries_time = forward_radon_freq(traces_matrix_prim, dt, h, q, 1, flow, fhigh);
primaries_multiples_time = forward_radon_freq(traces_matrix, dt, h, q, 1, flow, fhigh);
filtered_traces_time = forward_radon_freq(deconvolved_matrix, dt, h, q, 1, flow, fhigh);

%% Show reference trace and filtered traces in Time domain

figure(5)
imagesc(primaries_time, [-1 1]*1e-3)
title('Primaries')
axis([0 500 0 1000])
grid

figure(6)
imagesc(primaries_multiples_time, [-1 1]*1e-3)
title('Primaries and multiples')
axis([0 500 0 1000])
grid

figure(7)
imagesc(filtered_traces_time, [-1 1]*1e-3)
title('Filtered trace')
axis([0 500 0 1000])
grid

%% FIR filter

filter_one_len = 10;
prediction_step = 90;


%% Filtering all traces - FIR

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;

% Using cursor information
prediction_step = 0;
offset = 0;
regularization = 0;
filter_len = 6;

deconvolved_matrix = zeros(size(traces_matrix));
trace_nb = size(traces_matrix, 2);
mse_prediction = zeros(trace_nb, 1);
mse_reference_trace = zeros(trace_nb, 1);

for i=1:trace_nb
  
  % Nomalizing data
  [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, i, samples_start, attenuation_factor);
  trace_norm_prim = trace_pre_processing(traces_matrix_prim, i, samples_start, attenuation_factor);

  % Preparing data based in parameters
  % Using info from cursor
  prediction_step = predic_step(i)-offset;
  [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);
  gain = inv(train_set*train_set')*train_set*target'

  % Apply network to all traces
  deconvolved_matrix_fir(:, i) = target - gain'*train_set;
  mse_prediction_fir(i) = mean(deconvolved_matrix(:, i).^2);
  mse_reference_trace_fir(i) = mean((deconvolved_matrix_fir(:, i) - trace_norm_prim).^2);
  deconvolved_matrix_fir(:, i) = deconvolved_matrix_fir(:, i)*max_amp*std_dev + avg;

end

%% Show reference trace and filtered traces in Radon domain

figure(8)
imagesc(traces_matrix, [-1 1]*1e-5)
title('Primaries')
axis([0 500 0 1000])
grid

figure(9)
imagesc(deconvolved_matrix_fir, [-1 1]*1e-5)
title('Filtered trace')
axis([0 500 0 1000])
grid

%% Inverting traces with only primaries to use as reference

filtered_traces_time_fir = forward_radon_freq(deconvolved_matrix_fir, dt, h, q, 1, flow, fhigh);

%% Show reference trace and filtered traces in Time domain

figure(5)
imagesc(primaries_time, [-1 1]*1e-3)
title('Primaries')
axis([0 500 0 1000])
grid

figure(6)
imagesc(primaries_multiples_time, [-1 1]*1e-3)
title('Primaries and multiples')
axis([0 500 0 1000])
grid

figure(7)
imagesc(filtered_traces_time_fir, [-1 1]*1e-3)
title('Filtered trace')
axis([0 500 0 1000])
grid


%%
close all