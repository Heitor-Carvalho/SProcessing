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
load('CaseData1_0/tracos_in_time');
load('CaseData1_0/parameter');
load('CaseData1_0/full_radon_trace_offset_CaseData1_0.mat')

%% Case one primary and multiples

time = 0:dt:tmax;

% Ploting filtered trace and reference trace
traces_matrix = radon_p1_fst_mul_div_offset ;
traces_matrix_prim = radon_p1_fst_prim_div_offset;

figure(1)
imagesc(q, time, traces_matrix, [-1 1]*1e-6)
title('Radon - Primaries and multiples')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

figure(2)
imagesc(q, time, traces_matrix_prim, [-1 1]*1e-6)
title('Radon - Primaries')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

%% Primaries and multiples directly in time

% Primaries and multiples in time
traces_matrix_time = trace_p1_fst_prim_multiples;
traces_matrix_prim_time = trace_p1_fst_primaries;

figure(3)
imagesc(h, time, traces_matrix_prim_time, [-1 1]*5e-1)
title('Primaries in time')
axis([0 h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(4)
imagesc(h, time, traces_matrix_time, [-1 1]*5e-1)
title('Primaries and multiples in time')
axis([0 h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
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

%% Running ELM many times!
many = 0;

if(many)
  trace_nb = 22;
  n_times = 100;
  deconvolved_matrix = zeros(size(traces_matrix, 1), n_times);

  % Nomalizing data
  [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, i, samples_start, attenuation_factor);
  trace_norm_prim = trace_pre_processing(traces_matrix_prim, i, samples_start, attenuation_factor);

  for i=1:n_times

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
    prediction_step = predic_step(trace_nb);
    [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);

    % Calculating extreme learning machines values
    nn.w = calc_elm_weigths(train_set, target, regularization, nn)';

    % Apply network to all traces
    deconvolved_matrix(:, i) = (target - neural_nete(train_set, nn))';

  end
  
  elm_trace_std = std(deconvolved_matrix', 1);
end

%% Show reference trace and filtered traces in Radon domain

figure(5)
imagesc(q, time, traces_matrix, [-1 1]*1e-6)
title('Radon - Primaries and multiples')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

figure(6)
imagesc(q, time, deconvolved_matrix, [-1 1]*1e-6)
title('Filtered trace in radon domain - ELM')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

%% Inverting traces with only primaries to use as reference

min_offset_idx = 21;
primaries_time = forward_radon_freq(traces_matrix_prim, dt, h(min_offset_idx:end), q, 1, flow, fhigh);
primaries_multiples_time = forward_radon_freq(traces_matrix, dt, h(min_offset_idx:end), q, 1, flow, fhigh);
filtered_traces_time = forward_radon_freq(deconvolved_matrix, dt, h(min_offset_idx:end), q, 1, flow, fhigh);

%% Show reference trace and filtered traces in Time domain

figure(7)
imagesc(h(min_offset_idx:end), time, primaries_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon inverted primaries in time')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(8)
imagesc(h(min_offset_idx:end), time, primaries_multiples_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon inverted primaries and multiples in time')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(9)
imagesc(h(min_offset_idx:end), time, filtered_traces_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Inverted filtered trace in time domain - ELM')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Showing NMO corrected trace

p1_times = t0_p1:t0_water:tmax-t0_water;
v1 = 1500;
max_stretch = 35;

primaries_time_nmo = nmo(primaries_time(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times], [v1*ones(size(p1_times))], max_stretch);
primaries_multiples_time_nmo = nmo(primaries_multiples_time(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times], [v1*ones(size(p1_times))], max_stretch);
primaries_multiples_elm_time_nmo = nmo(primaries_multiples_time(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times], [v1*ones(size(p1_times))], max_stretch);
filtered_traces_elm_time_nmo = nmo(filtered_traces_time(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times], [v1*ones(size(p1_times))], max_stretch);

figure(10)
imagesc(h(min_offset_idx:end), time, primaries_time_nmo, [-1 1]*6e-4)
title('Radon inverted primaries NMO corrected')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(11)
imagesc(h(min_offset_idx:end), time, primaries_multiples_elm_time_nmo, [-1 1]*6e-4)
title('Radon inverted primaries and multiples NMO corrected')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(12)
imagesc(h(min_offset_idx:end), time, filtered_traces_elm_time_nmo, [-1 1]*6e-4)
title('Inverted filtered trace NMO corrected in time - ELM')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Filtering all traces - FIR

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;

% Using cursor information
prediction_step = 0;
offset = 0;
regularization = 1e-7;
filter_len = 15;

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
  mse_prediction_fir(i) = mean(deconvolved_matrix_fir(:, i).^2);
  mse_reference_trace_fir(i) = mean((deconvolved_matrix_fir(:, i) - trace_norm_prim).^2);
  deconvolved_matrix_fir(:, i) = deconvolved_matrix_fir(:, i)*max_amp*std_dev + avg;

end

%% Show reference trace and filtered traces in Radon domain

figure(13)
imagesc(q, time, traces_matrix, [-1 1]*1e-6)
title('Radon - Primaries and multiples')
axis([0 q(500) 0 time(1000)])
grid

figure(14)
imagesc(q, time, deconvolved_matrix_fir, [-1 1]*1e-6)
title('Inverted filtered trace in time domain - FIR')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

%% Inverting traces with only primaries to use as reference

filtered_traces_time_fir = forward_radon_freq(deconvolved_matrix_fir, dt, h(min_offset_idx:end), q, 1, flow, fhigh);

%% Show reference trace and filtered traces in Time domain

figure(15)
imagesc(h(min_offset_idx:end), time, primaries_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon inverted primaries in time')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(16)
imagesc(h(min_offset_idx:end), time, primaries_multiples_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon inverted primaries and multiples in time')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(17)
imagesc(h(min_offset_idx:end), time, filtered_traces_time_fir(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Inverted filtered trace in time domain - FIR')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Showing NMO corrected trace

filtered_traces_fir_time_nmo = nmo(filtered_traces_time_fir(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times], [v1*ones(size(p1_times))], 50);

figure(18)
imagesc(h(min_offset_idx:end), time, filtered_traces_fir_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Inverted filtered trace NMO corrected in time - FIR')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
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

%% Running ESN many times!

if(many)
  trace_nb = 22;
  n_times = 100;
  deconvolved_matrix_esn = zeros(size(traces_matrix, 1), n_times);

  % Nomalizing data
  [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, i, samples_start, attenuation_factor);
  trace_norm_prim = trace_pre_processing(traces_matrix_prim, i, samples_start, attenuation_factor);

  for i = 1:n_times
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
    prediction_step = predic_step(trace_nb);
    [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);

    % Calculating extreme learning machines values
    nn.w = calc_esn_weigths(train_set, target, regularization, nn);

    % Apply network to all traces
    deconvolved_matrix_esn(:, i) = target - neural_net_echo_states(train_set, nn);
  end
  
  esn_trace_std = std(deconvolved_matrix_esn', 1);
  
end

%% Showing ELM and ESN n_times execution std
if(many)
  figure(111)
  plot(time, esn_trace_std, 'b')
  hold on
  plot(time, elm_trace_std, 'k')
  xlabel('tau [s]')
  ylabel('Standard deviation')
  grid
end
%% Show reference trace and filtered traces in Radon domain

figure(19)
imagesc(q, time, traces_matrix, [-1 1]*1e-6)
title('Radon - Primaries and multiples')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

figure(20)
imagesc(q, time, deconvolved_matrix_esn, [-1 1]*1e-6)
title('Inverted filtered trace in time domain - ESN')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

%% Inverting traces with only primaries to use as reference

filtered_traces_time_esn = forward_radon_freq(deconvolved_matrix_esn, dt, h(min_offset_idx:end), q, 1, flow, fhigh);

%% Show reference trace and filtered traces in Time domain

figure(21)
imagesc(h(min_offset_idx:end), time, primaries_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon inverted primaries in time')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(22)
imagesc(h(min_offset_idx:end), time, primaries_multiples_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon inverted primaries and multiples in time')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(23)
imagesc(h(min_offset_idx:end), time, filtered_traces_time_esn(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Inverted filtered trace in time domain - ESN')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Showing NMO corrected trace

filtered_traces_esn_time_nmo = nmo(filtered_traces_time_esn(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times], [v1*ones(size(p1_times))], 50);

figure(24)
imagesc(h(min_offset_idx:end), time, filtered_traces_esn_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Inverted filtered trace NMO corrected in time - ESN')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Comparing FIR, ELM, ESN

figure(25)
imagesc(h(min_offset_idx:end), time, filtered_traces_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Filtered trace - ELM - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(26)
imagesc(h(min_offset_idx:end), time, filtered_traces_time_fir(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Filtered trace - FIR - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(27)
imagesc(h(min_offset_idx:end), time, filtered_traces_time_esn(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Filtered trace - ESN - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(28)
imagesc(h(min_offset_idx:end), time, primaries_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon - Only primaries - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Comparing FIR, ELM, ESN - NMO

figure(29)
imagesc(h(min_offset_idx:end), time, filtered_traces_elm_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Filtered trace - ELM - NMO - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(30)
imagesc(h(min_offset_idx:end), time, filtered_traces_fir_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Filtered trace - FIR - NMO - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(31)
imagesc(h(min_offset_idx:end), time, filtered_traces_esn_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Filtered trace - ESN - NMO - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(32)
imagesc(h(min_offset_idx:end), time, primaries_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon - Only primaries - NMO - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Comparing FIR, ELM, ESN - NMO - Stacked traces

stacking_bg = 1;
stacking_end = 45;

stacked_primaries_time_nmo = sum(primaries_time_nmo(:, stacking_bg:stacking_end), 2);
stacked_multiples_time_nmo = sum(primaries_multiples_time_nmo(:, stacking_bg:stacking_end), 2);
stacked_filtered_traces_fir_time_nmo = sum(filtered_traces_fir_time_nmo(:, stacking_bg:stacking_end), 2);
stacked_filtered_traces_esn_time_nmo = sum(filtered_traces_esn_time_nmo(:, stacking_bg:stacking_end), 2);
stacked_filtered_traces_elm_time_nmo = sum(filtered_traces_elm_time_nmo(:, stacking_bg:stacking_end), 2);

figure(33)
plot(time, stacked_primaries_time_nmo/max(stacked_primaries_time_nmo))
title('Radon - Only primaries stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(34)
plot(time, traces_matrix_prim_time(:, 1)/max(traces_matrix_time(:, 1)))
title('Only primaries stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(35)
plot(time, stacked_multiples_time_nmo/max(stacked_multiples_time_nmo))
title('Radon - Primaries and multiples stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(36)
plot(time, traces_matrix_time(:, 1)/max(traces_matrix_time(:, 1)))
title('Primaries and multiples stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(37)
plot(time, stacked_filtered_traces_fir_time_nmo/max(stacked_filtered_traces_fir_time_nmo))
title('Filtered trace - FIR - stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(38)
plot(time, stacked_filtered_traces_esn_time_nmo/max(stacked_filtered_traces_esn_time_nmo))
title('Filtered trace - ESN - stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(39)
plot(time, stacked_filtered_traces_elm_time_nmo/max(stacked_filtered_traces_elm_time_nmo))
title('Filtered trace - ELM - stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid


%% Stacked Section

figure(40)
imagesc(h(min_offset_idx:end), time, repmat(traces_matrix_prim_time(:, 1), 1, 500), [-1 1]*2e-1)
title('Only primaries stacked section')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(41)
imagesc(h(min_offset_idx:end), time, repmat(traces_matrix_time(:, 1), 1, 500), [-1 1]*2e-1)
title('Primaries and Multiples stacked section')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(42)
imagesc(h(min_offset_idx:end), time, repmat(stacked_primaries_time_nmo, 1, 500), [-1 1]*2e-2)
title('Radon - primaries stacked section')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(43)
imagesc(h(min_offset_idx:end), time, repmat(stacked_multiples_time_nmo, 1, 500), [-1 1]*2e-2)
title('Radon - primaries and multiples stacked section')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(44)
imagesc(h(min_offset_idx:end), time, repmat(stacked_filtered_traces_fir_time_nmo, 1, 500), [-1 1]*2e-2)
title('Filtered trace - FIR stacked section')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(45)
imagesc(h(min_offset_idx:end), time, repmat(stacked_filtered_traces_esn_time_nmo, 1, 500), [-1 1]*2e-2)
title('Filtered trace - ESN stacked section')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(46)
imagesc(h(min_offset_idx:end), time, repmat(stacked_filtered_traces_elm_time_nmo, 1, 500), [-1 1]*2e-2)
title('Filtered trace - ELM stacked section')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%%
% close all