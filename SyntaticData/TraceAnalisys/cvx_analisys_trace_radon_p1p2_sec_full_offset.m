%% Loading data

clear all

addpath('../../ThirdParty/SeismicLab/codes/radon_transforms/')
addpath('../../ThirdParty/SeismicLab/codes/velan_nmo/')
addpath('../../ThirdParty/SegyMAT/')
addpath('../../Tests');
addpath('../../../../../../../cvx/')
path = '../../SyntaticData/SimulatedDataGeneration/SynData_035/';
data_set_name = 'SynData_035';
load('SynData_035_offset_prediction_step.mat')
load([path, 'tracos_in_time.mat']);
load([path, 'tracos_in_radon']);
load([path, 'parameter']);

load_filtered = 1;
load_filtered_file = 'data_set035_filtered_offset.mat'

cvx_setup
cvx_quiet true

%% Case two primary and multiples - Zero offset

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

%% Filterin by quadratic programing optmization

traces_nb = size(traces_matrix, 2);
filter_one_len = 1;

train_matrix_filtered = zeros(size(traces_matrix));

windown_search = 10;
max_process_idx = 1000;
n = max_process_idx;
filtered_m = zeros(windown_search+1, length(traces_matrix));

if(load_filtered == 1)
  load(load_filtered_file);
else

  for i = 1:traces_nb

    hdiff = [1 -1];
    Hdiff = convmtx(hdiff, n);
    Hdiff(:, end) = [];

    prediction_steps = (prediction_step(i)-windown_search/2):(prediction_step(i)+windown_search/2);
  
    for j = 1:length(prediction_steps)

      gdiff = [1 zeros(1, prediction_steps(j)-1) -1];
      Gdiff = convmtx(gdiff, n);
      Gdiff(:, end-(prediction_steps(j)-1):end) = [];
    
      [trace_norm, avg, std_dev] = trace_normalization(traces_matrix(1:max_process_idx, i));
      [train_matrix, target] = trace_to_datatraining(trace_norm, filter_one_len, prediction_steps(j));

      train_matrix = train_matrix';
      target = target';
  
      % Also ommit the cvx log to go faser
      % Filtering using CVX
      cvx_begin
        variables x(n);
        minimize( norm(target - x.*train_matrix) );% + 0.001*norm(x'*Hdiff, 2) + 0.005*norm(x, inf) );
        subject to
          -5 <= x <= 1
  %        -0.4 <= x'*Gdiff <= 0.4
      cvx_end

      filtered_m(j, 1:max_process_idx) = target - x.*train_matrix;
  
    end
  
    traces_var = var(filtered_m');
    [min_var, min_idx] = min(traces_var);
  
    train_matrix_filtered(:, i) = filtered_m(min_idx, :);
    i
  
  end
  
end

train_matrix_filtered_med = medfilt1(train_matrix_filtered, 10);

%% Show reference trace and filtered traces in Radon domain

figure(5)
imagesc(q, time, traces_matrix, [-1 1]*1e-5)
title('Radon - Primaries and multiples')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

figure(6)
imagesc(q, time, train_matrix_filtered, [-1 1]*1e-0)
title('Filtered trace in radon domain - CVX - Primaries')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

figure(7)
imagesc(q, time, train_matrix_filtered_med, [-1 1]*1e-0)
title('Filtered trace in radon domain - CVX + Midfilt - Primaries')
axis([0 q(500) 0 time(1000)])
xlabel('p [s/m]')
ylabel('tau [s]')
grid

%% Inverting traces with only primaries to use as reference

min_offset_idx = 21;
primaries_time = forward_radon_freq(traces_matrix_prim, dt, h(min_offset_idx:end), q, 1, flow, fhigh);
primaries_multiples_time = forward_radon_freq(traces_matrix, dt, h(min_offset_idx:end), q, 1, flow, fhigh);
filtered_traces_time = forward_radon_freq(train_matrix_filtered, dt, h(min_offset_idx:end), q, 1, flow, fhigh);
filtered_traces_time_medfilt = forward_radon_freq(train_matrix_filtered_med, dt, h(min_offset_idx:end), q, 1, flow, fhigh);

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
imagesc(h(min_offset_idx:end), time, filtered_traces_time(:, 1:end-min_offset_idx), [-1 1]*8e1)
title('Inverted filtered trace in time domain - CVX Primaries')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(11)
imagesc(h(min_offset_idx:end), time, filtered_traces_time_medfilt(:, 1:end-min_offset_idx), [-1 1]*8e1)
title('Inverted filtered trace in time domain - CVX MedfiltPrimaries')
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
[filtered_traces_time_nmo, filtered_traces_time_nmo_muted] = nmo(filtered_traces_time(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);
[filtered_traces_time_nmo_medfilt, filtered_traces_time_nmo_medfilt_muted] = nmo(filtered_traces_time_medfilt(:, 1:end-min_offset_idx), dt, h(min_offset_idx:end), [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);

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
imagesc(h(min_offset_idx:end), time, filtered_traces_time_nmo, [-1 1]*9e1)
title('Inverted filtered trace NMO corrected in time - CVX Primaries')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

figure(15)
imagesc(h(min_offset_idx:end), time, filtered_traces_time_nmo_medfilt, [-1 1]*6e1)
title('Inverted filtered trace NMO corrected in time - CVX Medfilt Multiples')
axis([h(min_offset_idx) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Comparing CVX - NMO - Stacked traces

[ideal_primaries_in_time_nmo, ideal_primaries_in_time_nmo_muted] = nmo(traces_matrix_prim_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);
[ideal_multiples_in_time_nmo, ideal_multiples_in_time_nmo_muted] = nmo(traces_matrix_time, dt, h, [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], max_stretch);

stacked_ideal_primaries_time_nmo = zeros(length(time), 1);
stacked_ideal_multiples_time_nmo = zeros(length(time), 1);
stacked_primaries_time_nmo = zeros(length(time), 1);
stacked_multiples_time_nmo = zeros(length(time), 1);
stacked_filtered_traces_time_nmo = zeros(length(time), 1);
stacked_filtered_traces_time_nmo_medfilt = zeros(length(time), 1);

stacked_ideal_primaries_time(ideal_primaries_in_time_nmo_muted == 0) = 1;
stacked_ideal_multiples_time(ideal_multiples_in_time_nmo_muted == 0) = 1;
primaries_time_nmo_muted(primaries_time_nmo_muted == 0) = 1;
primaries_multiples_time_nmo_muted(primaries_multiples_time_nmo_muted == 0) = 1;
filtered_traces_time_nmo_muted(stacked_filtered_traces_time_nmo == 0) = 1;
filtered_traces_time_nmo_medfilt_muted(stacked_filtered_traces_time_nmo_medfilt == 0) = 1;

for i = 1:1500
  stacked_ideal_primaries_time_nmo(i) = mean(ideal_primaries_in_time_nmo(i, 1:ideal_primaries_in_time_nmo_muted(i)), 2);
  stacked_ideal_multiples_time_nmo(i) = mean(ideal_multiples_in_time_nmo(i, 1:ideal_multiples_in_time_nmo_muted(i)), 2);
  stacked_primaries_time_nmo(i) = mean(primaries_time_nmo(i, 1:primaries_time_nmo_muted(i)), 2);
  stacked_multiples_time_nmo(i) = mean(primaries_multiples_time_nmo(i, 1:primaries_multiples_time_nmo_muted(i)), 2);
  stacked_filtered_traces_time_nmo(i) = mean(filtered_traces_time_nmo(i, 1:filtered_traces_time_nmo_muted(i)), 2);
  stacked_filtered_traces_time_nmo_medfilt(i) = mean(filtered_traces_time_nmo_medfilt(i, 1:filtered_traces_time_nmo_medfilt_muted(i)), 2);
end

stacked_ideal_primaries_time_nmo = stacked_ideal_primaries_time_nmo*v1.*time';
stacked_ideal_multiples_time_nmo = stacked_ideal_multiples_time_nmo*v1.*time';
stacked_primaries_time_nmo = stacked_primaries_time_nmo*v1.*time';
stacked_multiples_time_nmo = stacked_multiples_time_nmo*v1.*time';
stacked_filtered_traces_time_nmo = stacked_filtered_traces_time_nmo*v1.*time';
stacked_filtered_traces_time_nmo_medfilt = stacked_filtered_traces_time_nmo_medfilt*v1.*time';

stacked_ideal_primaries_time_nmo_norm  = stacked_ideal_primaries_time_nmo/max(stacked_ideal_primaries_time_nmo);
stacked_ideal_multiples_time_nmo_norm  = stacked_ideal_multiples_time_nmo/max(stacked_ideal_multiples_time_nmo);
stacked_primaries_time_nmo_norm = stacked_primaries_time_nmo/max(stacked_primaries_time_nmo);
stacked_multiples_time_nmo_norm = stacked_multiples_time_nmo/max(stacked_multiples_time_nmo);
stacked_filtered_traces_time_nmo_norm = stacked_filtered_traces_time_nmo/max(stacked_filtered_traces_time_nmo);
stacked_filtered_traces_time_nmo_norm_medfilt = stacked_filtered_traces_time_nmo_medfilt/max(stacked_filtered_traces_time_nmo_medfilt);

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
plot(time, stacked_filtered_traces_time_nmo_norm)
title('Filtered trace - CVX - stacked trace - Cluster Primaries')
xlim([0 time(1500)])
xlabel('time [s]')
ylabel('Amplitude')
ylim([-1.5 1.5])
grid

figure(27)
plot(time, stacked_filtered_traces_time_nmo_norm_medfilt)
title('Filtered trace - CVX Medfilt - stacked trace - Cluster Multiples')
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
wiggle(time, 1:max_rep, repmat(stacked_filtered_traces_time_nmo_norm, 1, max_rep))
title('Filtered trace - CVX stacked section - Cluster Primaries')
ylim([0 time(800)])
ylabel('time [s]')
grid

figure(33)
wiggle(time, 1:max_rep, repmat(stacked_filtered_traces_time_nmo_norm_medfilt, 1, max_rep))
title('Filtered trace - CVX Medfilt stacked section - Cluster Multiples')
ylim([0 time(800)])
ylabel('time [s]')
grid

%%
close all
