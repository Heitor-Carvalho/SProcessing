%% NMO - Tests

clear all
close all

addpath('../../ThirdParty/SeismicLab/codes/velan_nmo/')
addpath('../../ThirdParty/SeismicLab/codes/radon_transforms/')

load('CaseData1_0/tracos_in_radon');
load('CaseData1_0/tracos_in_time');
load('CaseData1_0/parameter');
load('CaseData1_0/full_radon_trace_offset_CaseData1_0.mat')

%% Loading first primary only directly in time

traces_matrix_time = trace_p1p2_fst_primaries_div;

figure(1)
imagesc(traces_matrix_time, [-1 1]*6e-4)
title('Primaries 1 - Original trace in time')
xlim([0 500])
ylim([0 1000])
grid


p1_times = t0_p1:t0_water:tmax-t0_water;
p2_times = t0_p2:t0_water:tmax-t0_water;
v1 = 1500;
v2 = 1650;

figure(2)
imagesc(nmo(traces_matrix_time(:, 1:500), dt, h(1:500), [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], 50), [-1 1]*6e-4)
title('Primaries 1 - Original trace in time')
xlim([0 500])
ylim([0 1000])
grid


%% Doing the same with traces in time given by the inverse Radon

traces_matrix_prim = radon_p1p2_primaries_div;
traces_matrix_prim_offset = radon_p1p2_primaries_div_offset;

primaries_time = forward_radon_freq(traces_matrix_prim, dt, h, q, 1, flow, fhigh);
primaries_time_offset = forward_radon_freq(traces_matrix_prim_offset, dt, h, q, 1, flow, fhigh);

%% 

figure(3)
imagesc(primaries_time, [-1 1]*6e-4)
title('Primaries With offset 0 m')
xlim([0 500])
ylim([0 1000])
grid

figure(4)
imagesc(primaries_time_offset, [-1 1]*6e-4)
title('Primaries Without offset 0 m')
xlim([0 500])
ylim([0 1000])
grid

%%
figure(5)
imagesc(nmo(primaries_time(:, 1:300), dt, h(1:300), [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], 50), [-1 1]*6e-4)
grid

figure(6)
imagesc(nmo(primaries_time_offset(:, 1:300), dt, h(1:300), [p1_times, p2_times], [v1*ones(size(p1_times)), v2*ones(size(p1_times))], 50), [-1 1]*6e-4)
grid
