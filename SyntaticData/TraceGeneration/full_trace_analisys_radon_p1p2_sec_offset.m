%% Loading data

clear all
close all

addpath('../../Tests');
addpath('../../ThirdParty/SeismicLab/codes/radon_transforms/')
addpath('../../ThirdParty/SeismicLab/codes/velan_nmo/')
addpath('../../ThirdParty/SegyMAT/')
addpath('../../../IA353/NeuralNetwork/')
addpath('../../../IA353/ExtremeLearningMachine/')
addpath('../../../IA353/EchoStateNetworks/')
addpath('../../../IA353/Regularization/')

load('CaseData1_0/tracos_in_radon');
load('CaseData1_0/parameter');
load('CaseData1_0/full_radon_trace_offset_CaseData1_0.mat')

%% Case two primary and multiples

% Ploting filtered trace and reference trace
traces_matrix = radon_p1p2_sec_mul_div_offset;
traces_matrix_prim = radon_p1p2_primaries_div_offset;

figure(1)
imagesc(traces_matrix, [-1 1]*1e-6)
title('Primaries and multiples')
axis([0 500 0 1000])
grid

figure(2)
imagesc(traces_matrix_prim, [-1 1]*1e-6)
title('Primaries')
axis([0 500 0 1000])
grid

%% Filtering all traces - ELM

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;

% Using cursor information
prediction_step = 0;
offset = 0;
filter_len = 27;   
mid_layer_sz = 55;
regularization = 0;
initial_weigths_amp = 0.1;

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
  prediction_step = max(predic_step(i)-9, filter_len+1);
  [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);

  % Calculating extreme learning machines values
  nn.w = calc_elm_weigths(train_set, target, regularization, nn)';

  % Apply network to all traces
  deconvolved_matrix(:, i) = target - neural_nete(train_set, nn);
  mse_prediction(i) = mean(deconvolved_matrix(:, i).^2);
  mse_reference_trace(i) = mean((deconvolved_matrix(:, i) - trace_norm_prim).^2);
  deconvolved_matrix(:, i) = deconvolved_matrix(:, i)*max_amp*std_dev + avg;

end

% Show reference trace and filtered traces in Radon domain

figure(3)
imagesc(traces_matrix, [-1 1]*1e-6)
title('Primaries')
axis([0 500 0 1000])
grid

figure(4)
imagesc(deconvolved_matrix, [-1 1]*1e-6)
title('Filtered trace')
axis([0 500 0 1000])
grid

%% Inverting traces with only primaries to use as reference

primaries_time = forward_radon_freq(traces_matrix_prim, dt, h, q, 1, flow, fhigh);
primaries_multiples_time = forward_radon_freq(traces_matrix, dt, h, q, 1, flow, fhigh);
filtered_traces_time = forward_radon_freq(deconvolved_matrix, dt, h, q, 1, flow, fhigh);

%% Show reference trace and filtered traces in Time domain

figure(5)
imagesc(primaries_time, [-1 1]*6e-4)
title('Primaries')
axis([0 500 0 1000])
grid

figure(6)
imagesc(primaries_multiples_time, [-1 1]*6e-4)
title('Primaries and multiples')
axis([0 500 0 1000])
grid

figure(7)
imagesc(filtered_traces_time, [-1 1]*6e-4)
title('Filtered trace')
axis([0 500 0 1000])
grid

%% Showing NMO corrected trace

p1_times = t0_p1:t0_water:tmax-t0_water;
p2_times = t0_p2:t0_water:tmax-t0_water;
v1 = 1500;
v2 = 1650;

primaries_time_nmo = nmo(primaries_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], 50);
primaries_multiples_time_nmo = nmo(primaries_multiples_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], 50);
primaries_multiples_elm_time_nmo = nmo(primaries_multiples_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], 50);
filtered_traces_elm_time_nmo = nmo(filtered_traces_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], 50);

figure(8)
imagesc(primaries_time_nmo, [-1 1]*6e-4)
title('Only primaries NMO corrected')
axis([0 500 0 1000])
grid

figure(9)
imagesc(primaries_multiples_elm_time_nmo, [-1 1]*6e-4)
title('Primaries and multiples NMO corrected')
axis([0 500 0 1000])
grid

figure(10)
imagesc(filtered_traces_elm_time_nmo, [-1 1]*6e-4)
title('Filtered trace - ELM NMO corrected')
axis([0 500 0 1000])
grid

%% Filtering all traces - FIR

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;

% Using cursor information
prediction_step = 0;
offset = 0;
regularization = 1e-6;
filter_len = 8;

deconvolved_matrix_fir = zeros(size(traces_matrix));
trace_nb = size(traces_matrix, 2);
mse_prediction = zeros(trace_nb, 1);
mse_reference_trace = zeros(trace_nb, 1);

for i=1:trace_nb
  
  % Nomalizing data
  [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, i, samples_start, attenuation_factor);
  trace_norm_prim = trace_pre_processing(traces_matrix_prim, i, samples_start, attenuation_factor);

  % Preparing data based in parameters
  % Using info from cursor
  prediction_step = max(predic_step(i), filter_len+1);
  [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);
  gain = inv(train_set*train_set' + regularization*eye(size(train_set*train_set')))*train_set*target';

  % Apply network to all traces
  deconvolved_matrix_fir(:, i) = target - gain'*train_set;
  mse_prediction_fir(i) = mean(deconvolved_matrix(:, i).^2);
  mse_reference_trace_fir(i) = mean((deconvolved_matrix_fir(:, i) - trace_norm_prim).^2);
  deconvolved_matrix_fir(:, i) = deconvolved_matrix_fir(:, i)*max_amp*std_dev + avg;

end

%% Show reference trace and filtered traces in Radon domain

figure(11)
imagesc(traces_matrix, [-1 1]*1e-6)
title('Primaries')
axis([0 500 0 1000])
grid

figure(12)
imagesc(deconvolved_matrix_fir, [-1 1]*1e-6)
title('Filtered trace')
axis([0 500 0 1000])
grid

%% Inverting traces with only primaries to use as reference

filtered_traces_time_fir = forward_radon_freq(deconvolved_matrix_fir, dt, h, q, 1, flow, fhigh);

%% Show reference trace and filtered traces in Time domain

figure(13)
imagesc(primaries_time, [-1 1]*6e-4)
title('Primaries')
axis([0 500 0 1000])
grid

figure(14)
imagesc(primaries_multiples_time, [-1 1]*6e-4)
title('Primaries and multiples')
axis([0 500 0 1000])
grid

figure(15)
imagesc(filtered_traces_time_fir, [-1 1]*6e-4)
title('Filtered trace')
axis([0 500 0 1000])
grid

%% Showing NMO corrected trace

filtered_traces_fir_time_nmo = nmo(filtered_traces_time_fir, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], 50);

figure(16)
imagesc(filtered_traces_fir_time_nmo, [-1 1]*6e-4)
title('Filtered trace - FIR NMO corrected')
axis([0 500 0 1000])
grid

%% Filtering all traces - ESN

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;

% Using cursor information
prediction_step = 0;
offset = 0;
filter_len = 22;   
mid_layer_sz = 45;
regularization = 0;
initial_weigths_amp = 0.1;
spectral_radio = 0.1;

deconvolved_matrix_esn = zeros(size(traces_matrix));
trace_nb = size(traces_matrix, 2);
mse_prediction = zeros(trace_nb, 1);
mse_reference_trace = zeros(trace_nb, 1);

for i=1:trace_nb

  % Nomalizing data
  [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, i, samples_start, attenuation_factor);
  trace_norm_prim = trace_pre_processing(traces_matrix_prim, i, samples_start, attenuation_factor);

  % Neural network setup
  clear nn
  input_par.sz = [filter_len mid_layer_sz];
  input_par.range = initial_weigths_amp;
  feedback_par.sz = [mid_layer_sz mid_layer_sz];
  feedback_par.range = initial_weigths_amp;
  feedback_par.alpha = spectral_radio;
  out_sz = 1;
  nn.func = @tanh;
  nn.b = 0;

  [~, ~, W] = generate_echo_state_weigths(input_par, feedback_par);
  nn.v = W;
  nn = neuro_net_init(nn);

  % Preparing data based in parameters
  % Using info from cursor
  prediction_step = max(predic_step(i)-9, filter_len+1);
  [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);

  % Calculating extreme learning machines values
  nn.w = calc_esn_weigths(train_set, target, regularization, nn);

  % Apply network to all traces
  deconvolved_matrix_esn(:, i) = target - neural_net_echo_states(train_set, nn);
  mse_prediction(i) = mean(deconvolved_matrix_esn(:, i).^2);
  mse_reference_trace(i) = mean((deconvolved_matrix_esn(:, i) - trace_norm_prim).^2);
  deconvolved_matrix_esn(:, i) = deconvolved_matrix_esn(:, i)*max_amp*std_dev + avg;

end

% Show reference trace and filtered traces in Radon domain

figure(17)
imagesc(traces_matrix, [-1 1]*1e-6)
title('Primaries')
axis([0 500 0 1000])
grid

figure(18)
imagesc(deconvolved_matrix_esn, [-1 1]*1e-6)
title('Filtered trace')
axis([0 500 0 1000])
grid

%% Inverting traces with only primaries to use as reference

filtered_traces_time_esn = forward_radon_freq(deconvolved_matrix_esn, dt, h, q, 1, flow, fhigh);

%% Show reference trace and filtered traces in Time domain

figure(15)
imagesc(primaries_time, [-1 1]*6e-4)
title('Primaries')
axis([0 500 0 1000])
grid

figure(16)
imagesc(primaries_multiples_time, [-1 1]*6e-4)
title('Primaries and multiples')
axis([0 500 0 1000])
grid

figure(17)
imagesc(filtered_traces_time_esn, [-1 1]*6e-4)
title('Filtered trace')
axis([0 500 0 1000])
grid

%% Showing NMO corrected trace

filtered_traces_esn_time_nmo = nmo(filtered_traces_time_esn, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], 50);

figure(19)
imagesc(filtered_traces_esn_time_nmo, [-1 1]*6e-4)
title('Filtered trace - ESN NMO corrected')
axis([0 500 0 1000])
grid

%% Comparing FIR, ELM, ESN

figure(20)
imagesc(filtered_traces_time, [-1 1]*6e-4)
title('Filtered trace - ELM')
axis([0 500 0 1000])
grid

figure(21)
imagesc(filtered_traces_time_fir, [-1 1]*6e-4)
title('Filtered trace - FIR')
axis([0 500 0 1000])
grid

figure(22)
imagesc(filtered_traces_time_esn, [-1 1]*6e-4)
title('Filtered trace - ESN')
axis([0 500 0 1000])
grid

figure(23)
imagesc(primaries_time, [-1 1]*6e-4)
title('Only primaries')
axis([0 500 0 1000])
grid

%% Comparing FIR, ELM, ESN - NMO

figure(24)
imagesc(filtered_traces_elm_time_nmo, [-1 1]*6e-4)
title('Filtered trace - ELM')
axis([0 500 0 1000])
grid

figure(25)
imagesc(filtered_traces_fir_time_nmo, [-1 1]*6e-4)
title('Filtered trace - FIR')
axis([0 500 0 1000])
grid

figure(26)
imagesc(filtered_traces_esn_time_nmo, [-1 1]*6e-4)
title('Filtered trace - ESN')
axis([0 500 0 1000])
grid

figure(27)
imagesc(primaries_time_nmo, [-1 1]*6e-4)
title('Only primaries')
axis([0 500 0 1000])
grid


%% Comparing FIR, ELM, ESN - NMO - Stacked traces

stacking_bg = 19;
stacking_end = 101;

figure(28)
plot(sum(primaries_time_nmo(:, stacking_bg:stacking_end), 2))
title('Only primaries')
xlim([0 1500])
grid

figure(29)
plot(sum(primaries_multiples_time_nmo(:, stacking_bg:stacking_end), 2))
title('Primaries and multiples')
xlim([0 1500])
grid

figure(30)
plot(sum(filtered_traces_fir_time_nmo(:, stacking_bg:stacking_end), 2))
title('Filtered trace - FIR')
xlim([0 1500])
grid

figure(31)
plot(sum(filtered_traces_esn_time_nmo(:, stacking_bg:stacking_end), 2))
title('Filtered trace - ESN')
xlim([0 1500])
grid

figure(32)
plot(sum(filtered_traces_elm_time_nmo(:, stacking_bg:stacking_end), 2))
title('Filtered trace - ELM')
xlim([0 1500])
grid

%%
close all