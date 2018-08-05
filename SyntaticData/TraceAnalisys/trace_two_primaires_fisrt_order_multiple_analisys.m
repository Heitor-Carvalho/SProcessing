%% Loading data

addpath('../../Tests');

load('CaseData1_0/tracos_in_time_ideal');
load('CaseData1_0/parameter');

%% Case 5.0 - Anlisys of time trace - Two primary and multiples - One generating system

time = 0:dt:tmax;

% Plotting the trace
trace_1 = trace_p1p2_fst_prim_multiples_time(:, 1);
trace_p = trace_p1p2_fst_primaries_time(:, 1);

figure(1)
plot(time, trace_1)
hold on
plot(time, trace_p, '--r')
legend('Primaires + Multiples', 'Primaries')
xlim([0 1.5])
grid

%% Getting the traning data (Matrix used in regression)

% In this simple case, a filter with one coefficient is enough to recover 
% the primaries

filter_one_len = 1;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(trace_1, filter_one_len, prediction_step);

gain = inv(train_matrix*train_matrix')*train_matrix*target'

figure(2)
plot(train_matrix, target,'.', train_matrix, train_matrix*gain)
legend('Function to be approximated', 'FIR with one delay aproximation')
grid

figure(3)
plot(time, target, time, target - train_matrix*gain, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

%% Now, let's add some noise and check this again!
trace_1_noisy = trace_1 + 0.02*randn(size(trace_1));

% First, lets look the noise trace
figure(5)
plot(time, trace_1_noisy)
grid

filter_one_len = 1;
[train_matrix_noisy, target_noisy] = trace_to_datatraining(trace_1_noisy, filter_one_len, prediction_step);

gain_noisy = inv(train_matrix_noisy*train_matrix_noisy')*train_matrix_noisy*target_noisy'

figure(6)
plot(train_matrix_noisy, target_noisy,'.', train_matrix_noisy, train_matrix_noisy*gain_noisy)
legend('Function to be approximated', 'FIR with one delay aproximation')
grid

figure(7)
plot(time, target_noisy, time, target_noisy - train_matrix_noisy*gain_noisy, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

% Now the noisy gain is not exactly 0.6 and we can see a cloud of points around
% the zero in the graph. 

%% Removing points with low energy

% We can try to eliminate the points with low energy, this way we can reduce
% the noise and the outlier points.

noisy_trace_target_energy = target_noisy.^2;

% Plotting energy time curve
figure(5)
plot(time, noisy_trace_target_energy)
legend('Trace energy')
grid

% Chosing a threshold and removing all points with energy below,
% we've got:
energy_threshold = 0.01;
removed_idx = mod(noisy_trace_target_energy < energy_threshold, 2);

gain_noisy_less_points = inv(train_matrix_noisy(~removed_idx)*train_matrix_noisy(~removed_idx)')*train_matrix_noisy(~removed_idx)*target_noisy(~removed_idx)'

figure(8)
plot(train_matrix_noisy(~removed_idx), target_noisy(~removed_idx),'.', train_matrix_noisy(~removed_idx), train_matrix_noisy(~removed_idx)*gain_noisy)
legend('Function to be mapped points', 'FIR with one delay aproximation')
grid

figure(9)
plot(time, target_noisy, time, target_noisy - train_matrix_noisy*gain_noisy_less_points, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

% As can be seen, there was some improvment !!

%% We can also see the regression when the filter has length two

filter_one_len = 2;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(trace_1, filter_one_len, prediction_step);

gain = inv(train_matrix*train_matrix')*train_matrix*target'

[mesh_x, mesh_y] = meshgrid(-1:0.1:1, -1:0.1:1);
regression_plan = [mesh_x(:), mesh_y(:)]*gain;
regression_plan = reshape(regression_plan, size(mesh_x));

% Regression for filter length 2
figure(10)
plot3(train_matrix(1, :), train_matrix(2, :), target,'.', train_matrix(1,:), train_matrix(2,:), gain'*train_matrix)
hold on
mesh(mesh_x, mesh_y, regression_plan)
view(30, 18);
grid

figure(11)
plot(time, target, time, target - gain'*train_matrix, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid
%% Fitting a Gaussian Mixture Model
rng(10);
end_process = 801;

filter_one_len = 1;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(trace_1, filter_one_len, prediction_step);

data_set = [train_matrix; target]';
data_set(end_process:end, :) = [];

init_gues = 2*ones(size(data_set,1), 1);
% Initializing first cluster with samples from the first primary
init_gues(1:150) = 1; 
% Initializing second cluster with all other samples
init_gues(151:end) = 2;

gm = fitgmdist(data_set, 2, 'Start', init_gues, 'RegularizationValue', 1e-9, 'CovarianceType', 'Diagonal');
posterior = gm.posterior(data_set);

figure(12)
h = ezcontour(@(x,y)pdf(gm,[x y]),[-1 1],[-1 1], 500);
hold on
plot(train_matrix(1:end_process), target(1:end_process),'.')
grid

figure(13)
plot(target(1:800))
hold on
plot(posterior(:, 1))
legend('Trace', 'Probability of been a primary')
grid

figure(14)
plot(target(1:800))
hold on
plot(posterior(:, 2))
legend('Trace', 'Probability of been a multiple')
grid

%% Filtered trace using decision limit at 50%

tg = target(1:800);
tg(posterior(:, 1) < 0.5) = 0;
figure(15)
plot(tg)
legend('Trace with probability of primarie higher than 0.5 - Primary')
grid

tg = target(1:800);
tg(posterior(:, 2) < 0.5) = 0;
figure(16)
plot(tg)
legend('Trace with probability of primarie muliple than 0.5 - Multiple')
grid


%%
close all