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
traces_matrix_time = trace_p1p2_sec_prim_multiples_div(:, 31:end);
traces_matrix_prim_time = trace_p1p2_fst_primaries_div(:, 31:end);

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
axis([h(1) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

traces_matrix_time_correted = traces_matrix_time*(v1(1)+v2(1))/2.*repmat(time', 1, length(h(31:end)));

figure(5)
imagesc(h, time, traces_matrix_time_correted, [-1 1]*5e-1)
title('Primaries and multiples in time')
axis([h(1) h(500) 0 time(1000)])
xlabel('offset [m]')
ylabel('time [s]')
grid

%% Velocity Analisis
[Snorm, S, S1, S2, tau, v] = velan(traces_matrix_time_correted(1:800, :), dt, h(31:end), 1400, 1700, 400, 1, 15);

imagesc(v, tau, S, [0 1]*1e5)

%% Normalizing - By line
norm_factor = sum(Snorm,2);
S_norm = S./repmat(norm_factor, 1, size(S,2));

imagesc(v, tau, S_norm , [0 1]*5e3)
%%
grid
plot(tau, S(:, 134))
%%
plot(tau, S(:, 334))
grid

%% Velocity spectrum - Sum of energy for all velocities
plot(tau, sum(S,2))
grid
%%
plot(v, log(sum(S,1)))
