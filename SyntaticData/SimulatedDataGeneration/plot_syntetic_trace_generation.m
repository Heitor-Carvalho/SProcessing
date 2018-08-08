clear all;
close all;

addpath('../../ThirdParty/SegyMAT/')
load('./SynData_025/parameter.mat')
load('./SynData_025/tracos_in_radon.mat')
load('./SynData_025/tracos_in_time.mat')

time_idx_plot = 1:2:900;
trace_idx_plot = 1:2:300;
time_idx = 0:dt:tmax;
time_idx = time_idx(time_idx_plot);

%% All cases: traces in time domain

% Case 1 - Only P1
figure(1)
wiggle(time_idx, trace_idx_plot, trace_p1_fst_primaries(time_idx_plot, trace_idx_plot))
title('Primaries 1')
xlim([0 max(trace_idx_plot)])
ylim([0 max(time_idx)])
grid

% Case 2 - P1 and fisrt order multiples

figure(2)
wiggle(time_idx, trace_idx_plot, trace_p1_fst_prim_multiples(time_idx_plot, trace_idx_plot))
title('Primaries 1 and fisrt order multiples')
xlim([0 max(trace_idx_plot)])
ylim([0 max(time_idx)])
grid

% Case 3 - Only P2
figure(3)
wiggle(time_idx, trace_idx_plot, trace_p2_fst_primaries(time_idx_plot, trace_idx_plot))
title('Primaries 2')
xlim([0 max(trace_idx_plot)])
ylim([0 max(time_idx)])
grid

% Case 4 - P2 and fisrt order multiples

figure(4)
wiggle(time_idx, trace_idx_plot, trace_p2_fst_prim_multiples(time_idx_plot, trace_idx_plot))
title('Primaries 2 and fisrt order multiples')
xlim([0 max(trace_idx_plot)])
ylim([0 max(time_idx)])
grid

% Case 5 - P1 and P2 fisrt order multiples

figure(5)
wiggle(time_idx, trace_idx_plot, trace_p1p2_fst_prim_multiples(time_idx_plot, trace_idx_plot))
title('Primaries 1, 2 and fisrt order multiples')
xlim([0 max(trace_idx_plot)])
ylim([0 max(time_idx)])
grid

% Case 6 - P2 second order multiples

figure(6)
wiggle(time_idx, trace_idx_plot, trace_p2_fst_prim_multiples(time_idx_plot, trace_idx_plot))
title('Primaries 2 and second order multiples')
xlim([0 max(trace_idx_plot)])
ylim([0 max(time_idx)])
grid

% Case 7 - P1 and P2 second order multiples

figure(7)
wiggle(time_idx, trace_idx_plot, trace_p1p2_sec_prim_multiples(time_idx_plot, trace_idx_plot))
title('Primaries 1, 2 and second order multiples')
xlim([0 max(trace_idx_plot)])
ylim([0 max(time_idx)])
grid

% Case 8 - P1 and P2 only primaries

figure(8)
wiggle(time_idx, trace_idx_plot, trace_p1p2_fst_primaries(time_idx_plot, trace_idx_plot))
title('Primaries 1, 2 primaries')
xlim([0 max(trace_idx_plot)])
ylim([0 max(time_idx)])
grid

%% All cases: traces in radon domain

figure(9)
wiggle(radon_p1_fst_prim_div(time_idx_plot, trace_idx_plot))
title('Primaries 1 - Radon Domain')
xlim([0 length(trace_idx_plot)])
ylim([0 length(time_idx_plot)])
grid

% Case 2 - P1 and fisrt order multiples

figure(10)
wiggle(radon_p1_fst_mul_div(time_idx_plot, trace_idx_plot))
title('Primaries 1 and first order multiples - Radon Domain')
xlim([0 length(trace_idx_plot)])
ylim([0 length(time_idx_plot)])
grid

% Case 3 - Only P2

figure(11)
wiggle(radon_p2_fst_prim_div(time_idx_plot, trace_idx_plot))
title('Primaries 2 - Radon Domain')
xlim([0 length(trace_idx_plot)])
ylim([0 length(time_idx_plot)])
grid

% Case 4 - P2 and fisrt order multiples

figure(12)
wiggle(radon_p2_fst_mul_div(time_idx_plot, trace_idx_plot))
title('Primaries 2 and fisrt order multiples - Radon Domain')
xlim([0 length(trace_idx_plot)])
ylim([0 length(time_idx_plot)])
grid

% Case 5 - P1 and P2 fisrt order multiples

figure(13)
wiggle(radon_p1p2_fst_mul_div(time_idx_plot, trace_idx_plot))
title('P1 and P2 fisrt order multiples - Radon Domain')
xlim([0 length(trace_idx_plot)])
ylim([0 length(time_idx_plot)])
grid

% Case 6 - P2 second order multiples

figure(14)
wiggle(radon_p2_sec_mul_div(time_idx_plot, trace_idx_plot))
title('P2 second order multiples - Radon Domain')
xlim([0 length(trace_idx_plot)])
ylim([0 length(time_idx_plot)])
grid

% Case 7 - P1 and P2 second order multiples

figure(15)
wiggle(radon_p1p2_sec_mul_div(time_idx_plot, trace_idx_plot))
title('P1 and P2 second order multiples - Radon Domain')
xlim([0 length(trace_idx_plot)])
ylim([0 length(time_idx_plot)])
grid

% Case 8 - P1 and P2 primries

figure(21)
wiggle(radon_p1p2_primaries_div(time_idx_plot, trace_idx_plot))
title('P1 and P2 primaries - Radon Domain')
xlim([0 length(trace_idx_plot)])
ylim([0 length(time_idx_plot)])
grid

%%
close all
