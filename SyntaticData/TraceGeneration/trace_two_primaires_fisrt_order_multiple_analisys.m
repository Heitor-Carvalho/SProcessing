%% Loading data

addpath('../../Tests');

load('CaseData1_0/tracos_in_time_ideal');
load('CaseData1_0/parameter');

%% Case 5.0 - Anlisys of time trace - Two primary and multiples

time = 0:dt:tmax;

% Plotting the trace
trace_1 = trace_p1p2_fst_prim_multiples_time(:, 1);

figure(1)
plot(time, trace_1)
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

%%
close all