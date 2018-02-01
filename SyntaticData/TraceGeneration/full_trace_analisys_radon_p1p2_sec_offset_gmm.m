%% Loading data

clear all
close all

addpath('../../Tests');
addpath('../../ThirdParty/SeismicLab/codes/radon_transforms/')
addpath('../../ThirdParty/SeismicLab/codes/velan_nmo/')
addpath('../../ThirdParty/SegyMAT/')
addpath('../../../IA353/ExtremeLearningMachine/')
addpath('../../../IA353/Regularization/')

load('CaseData1_0/tracos_in_radon');
load('CaseData1_0/tracos_in_time');
load('CaseData1_0/parameter');
load('CaseData1_0/full_radon_trace_offset_gmm_CaseData1_0.mat')
load('CaseData1_0/full_radon_trace_offset_CaseData1_0.mat')

%% Case two primary and multiples

time = 0:dt:tmax;

% Ploting filtered trace and reference trace
traces_matrix = radon_p1p2_sec_mul_div_offset;
traces_matrix_prim = radon_p1p2_primaries_div_offset;

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
traces_matrix_time = trace_p1p2_sec_prim_multiples_div;
traces_matrix_prim_time = trace_p1p2_fst_primaries_div;

figure(3)
imagesc(h, time, traces_matrix_prim_time, [-1 1]*5e-4)
title('Primaries in time')
axis([0 h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(4)
imagesc(h, time, traces_matrix_time, [-1 1]*5e-4)
title('Primaries and multiples in time')
axis([0 h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Filtering all traces - GMM

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;

% Using cursor information
prediction_step = 0;
offset = 0;
filter_len = 22;
regularization = 1e-5;
primary_len = 70;
primaries_gmm_matrix = zeros(size(traces_matrix));
multiples_gmm_matrix = zeros(size(traces_matrix));
trace_nb = size(traces_matrix, 2);
max_process = 700;
options = statset('MaxIter', 1000);

for i=1:trace_nb

  % Nomalizing data
  [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, i, samples_start, attenuation_factor);
  trace_norm_prim = trace_pre_processing(traces_matrix_prim, i, samples_start, attenuation_factor);

  prediction_step = max(predic_step(i), filter_len+1);
  [train_matrix, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);

  % Fitting a GMM
  clear gm
  
  prim_idx = max(prim_center_idx(i)-round(primary_len/2), 1):(prim_center_idx(i)+round(primary_len/2)); 
 
  init_gues(prim_idx) = 1;
  init_gues(setxor(1:length(trace_norm), prim_idx)) = 2; 
  
  data_set = [train_matrix', target'];
 
  gm = fitgmdist(data_set(1:max_process, :), 2, 'Start', init_gues(1:max_process), 'RegularizationValue', regularization, 'CovarianceType', 'Diagonal', 'Options', options);
  posterior = gm.posterior([train_matrix', target']);
 
  % Filtering traces with 50 % probability threshold
  primaries_gmm_matrix(:, i) = target; 
  multiples_gmm_matrix(:, i) = target; 
  primaries_gmm_matrix(posterior(:, 2) > 0.5, i) = 0;
  multiples_gmm_matrix(posterior(:, 2) <= 0.5, i) = 0;
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
imagesc(q, time, primaries_gmm_matrix, [-1 1]*1e-1)
title('Filtered trace in radon domain - GMM - Primaries')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

figure(7)
imagesc(q, time, multiples_gmm_matrix, [-1 1]*1e-1)
title('Filtered trace in radon domain - GMM - Multiples')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

%% Inverting traces with only primaries to use as reference

min_offset_idx = 21;
primaries_time = forward_radon_freq(traces_matrix_prim, dt, h(min_offset_idx:end), q, 1, flow, fhigh);
primaries_multiples_time = forward_radon_freq(traces_matrix, dt, h(min_offset_idx:end), q, 1, flow, fhigh);
filtered_traces_time = forward_radon_freq(primaries_gmm_matrix, dt, h(min_offset_idx:end), q, 1, flow, fhigh);
filtered_traces_time_mult = forward_radon_freq(multiples_gmm_matrix, dt, h(min_offset_idx:end), q, 1, flow, fhigh);

%% Show reference trace and filtered traces in Time domain

figure(8)
imagesc(h(min_offset_idx:end), time, primaries_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon inverted primaries in time')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(9)
imagesc(h(min_offset_idx:end), time, primaries_multiples_time(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon inverted primaries and multiples in time')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(10)
imagesc(h(min_offset_idx:end), time, filtered_traces_time(:, 1:end-min_offset_idx), [-1 1]*5e1)
title('Inverted filtered trace in time domain - GMM - Cluster Primaries')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(11)
imagesc(h(min_offset_idx:end), time, filtered_traces_time_mult(:, 1:end-min_offset_idx), [-1 1]*5e1)
title('Inverted filtered trace in time domain - GMM - Cluster Multiples')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Showing NMO corrected trace

p1_times = t0_p1:t0_water:tmax-t0_water;
p2_times = t0_p2:t0_water:tmax-t0_water;
v1 = 1500;
v2 = 1650;
max_stretch = 35;

[primaries_time_nmo, primaries_time_nmo_muted] = nmo(primaries_time(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);
[primaries_multiples_time_nmo, primaries_multiples_time_nmo_muted] = nmo(primaries_multiples_time(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);
[filtered_traces_gmm_time_nmo, filtered_traces_gmm_time_nmo_muted] = nmo(filtered_traces_time(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);
[filtered_traces_mult_gmm_time_nmo, filtered_traces_mult_gmm_time_nmo_muted] = nmo(filtered_traces_time_mult(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);

figure(12)
imagesc(h(min_offset_idx:end), time, primaries_time_nmo, [-1 1]*6e-4)
title('Radon inverted primaries NMO corrected')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(13)
imagesc(h(min_offset_idx:end), time, primaries_multiples_time_nmo, [-1 1]*6e-4)
title('Radon inverted primaries and multiples NMO corrected')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(14)
imagesc(h(min_offset_idx:end), time, filtered_traces_gmm_time_nmo, [-1 1]*6e1)
title('Inverted filtered trace NMO corrected in time - GMM - Cluster Primaries')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(15)
imagesc(h(min_offset_idx:end), time, filtered_traces_mult_gmm_time_nmo, [-1 1]*6e1)
title('Inverted filtered trace NMO corrected in time - GMM - Cluster Multiples')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Comparing GMM and Original

% NMO correting and stacking for ideal and original data in time!
[ideal_primaries_in_time_nmo, ideal_primaries_in_time_nmo_muted] = nmo(traces_matrix_prim_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);
[ideal_multiples_in_time_nmo, ideal_multiples_in_time_nmo_muted] = nmo(traces_matrix_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);

figure(18)
imagesc(h(min_offset_idx:end), time, filtered_traces_gmm_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e1)
title('Filtered trace - GMM - NMO - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(19)
imagesc(h(min_offset_idx:end), time, primaries_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Radon - Only primaries - NMO - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(20)
imagesc(h(min_offset_idx:end), time, ideal_primaries_in_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Only primaries - NMO - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(21)
imagesc(h(min_offset_idx:end), time, ideal_multiples_in_time_nmo(:, 1:end-min_offset_idx), [-1 1]*6e-4)
title('Primaries and Multiples - NMO - Comp')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Comparing GMM - NMO - Stacked traces

[ideal_primaries_in_time_nmo, ideal_primaries_in_time_nmo_muted] = nmo(traces_matrix_prim_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);
[ideal_multiples_in_time_nmo, ideal_multiples_in_time_nmo_muted] = nmo(traces_matrix_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);

stacked_ideal_primaries_time_nmo = zeros(length(time), 1);
stacked_ideal_multiples_time_nmo = zeros(length(time), 1);
stacked_primaries_time_nmo = zeros(length(time), 1);
stacked_multiples_time_nmo = zeros(length(time), 1);
stacked_filtered_traces_gmm_time_nmo = zeros(length(time), 1);
stacked_filtered_traces_mult_gmm_time_nmo = zeros(length(time), 1);

stacked_ideal_primaries_time(ideal_primaries_in_time_nmo_muted == 0) = 1;
stacked_ideal_multiples_time(ideal_multiples_in_time_nmo_muted == 0) = 1;
primaries_time_nmo_muted(primaries_time_nmo_muted == 0) = 1;
primaries_multiples_time_nmo_muted(primaries_multiples_time_nmo_muted == 0) = 1;
filtered_traces_gmm_time_nmo_muted(filtered_traces_gmm_time_nmo_muted == 0) = 1;
filtered_traces_mult_gmm_time_nmo_muted(filtered_traces_mult_gmm_time_nmo_muted == 0) = 1;

for i = 1:1500
  stacked_ideal_primaries_time_nmo(i) = mean(ideal_primaries_in_time_nmo(i, 1:ideal_primaries_in_time_nmo_muted(i)), 2);
  stacked_ideal_multiples_time_nmo(i) = mean(ideal_multiples_in_time_nmo(i, 1:ideal_multiples_in_time_nmo_muted(i)), 2);
  stacked_primaries_time_nmo(i) = mean(primaries_time_nmo(i, 1:primaries_time_nmo_muted(i)), 2);
  stacked_multiples_time_nmo(i) = mean(primaries_multiples_time_nmo(i, 1:primaries_multiples_time_nmo_muted(i)), 2);
  stacked_filtered_traces_gmm_time_nmo(i) = mean(filtered_traces_gmm_time_nmo(i, 1:filtered_traces_gmm_time_nmo_muted(i)), 2);
  stacked_filtered_traces_mult_gmm_time_nmo(i) = mean(filtered_traces_mult_gmm_time_nmo(i, 1:filtered_traces_mult_gmm_time_nmo_muted(i)), 2);
end

stacked_ideal_primaries_time_nmo = stacked_ideal_primaries_time_nmo*v1.*time';
stacked_ideal_multiples_time_nmo = stacked_ideal_multiples_time_nmo*v1.*time';
stacked_primaries_time_nmo = stacked_primaries_time_nmo*v1.*time';
stacked_multiples_time_nmo = stacked_multiples_time_nmo*v1.*time';
stacked_filtered_traces_gmm_time_nmo = stacked_filtered_traces_gmm_time_nmo*v1.*time';
stacked_filtered_traces_mult_gmm_time_nmo = stacked_filtered_traces_mult_gmm_time_nmo*v1.*time';

stacked_ideal_primaries_time_nmo_norm  = stacked_ideal_primaries_time_nmo/max(stacked_ideal_primaries_time_nmo);
stacked_ideal_multiples_time_nmo_norm  = stacked_ideal_multiples_time_nmo/max(stacked_ideal_multiples_time_nmo);
stacked_primaries_time_nmo_norm = stacked_primaries_time_nmo/max(stacked_primaries_time_nmo);
stacked_multiples_time_nmo_norm = stacked_multiples_time_nmo/max(stacked_multiples_time_nmo);
stacked_filtered_traces_gmm_time_nmo_norm = stacked_filtered_traces_gmm_time_nmo/max(stacked_filtered_traces_gmm_time_nmo);
stacked_filtered_traces_mult_gmm_time_nmo_norm = stacked_filtered_traces_mult_gmm_time_nmo/max(stacked_filtered_traces_mult_gmm_time_nmo);

figure(22)
plot(time, stacked_primaries_time_nmo_norm)
title('Radon - Only primaries stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(23)
plot(time, stacked_ideal_primaries_time_nmo_norm)
title('Only primaries stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(24)
plot(time, stacked_multiples_time_nmo_norm)
title('Radon - Primaries and multiples stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(25)
plot(time, stacked_ideal_multiples_time_nmo_norm)
title('Primaries and multiples stacked trace')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
grid

figure(26)
plot(time, stacked_filtered_traces_gmm_time_nmo_norm)
title('Filtered trace - GMM - stacked trace - Cluster Primaries')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
ylim([-1.5 1.5])
grid

figure(27)
plot(time, stacked_filtered_traces_mult_gmm_time_nmo_norm)
title('Filtered trace - GMM - stacked trace - Cluster Multiples')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
ylim([-1.5 1.5])
grid


%% Stacked Section

max_rep = 30;

figure(28)
wiggle(time, 1:max_rep, repmat(stacked_ideal_primaries_time_nmo_norm, 1, max_rep))
title('Only primaries stacked section')
ylim([0 time(800)])
ylabel('time [s]')
grid

figure(29)
wiggle(time, 1:max_rep, repmat(stacked_ideal_multiples_time_nmo_norm, 1, max_rep))
title('Primaries and Multiples stacked section')
ylim([0 time(800)])
ylabel('time [s]')
grid

figure(30)
wiggle(time, 1:max_rep, repmat(stacked_primaries_time_nmo_norm, 1, max_rep))
title('Radon - primaries stacked section')
ylim([0 time(800)])
ylabel('time [s]')
grid

figure(31)
wiggle(time, 1:max_rep, repmat(stacked_multiples_time_nmo_norm, 1, max_rep))
title('Radon - primaries and multiples stacked section')
ylim([0 time(800)])
ylabel('time [s]')
grid

figure(32)
wiggle(time, 1:max_rep, repmat(stacked_filtered_traces_gmm_time_nmo_norm, 1, max_rep))
title('Filtered trace - GMM stacked section - Cluster Primaries')
ylim([0 time(800)])
ylabel('time [s]')
grid

figure(33)
wiggle(time, 1:max_rep, repmat(stacked_filtered_traces_mult_gmm_time_nmo_norm, 1, max_rep))
title('Filtered trace - GMM stacked section - Cluster Multiples')
ylim([0 time(800)])
ylabel('time [s]')
grid

%%
close all