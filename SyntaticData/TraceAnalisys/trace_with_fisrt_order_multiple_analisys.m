%% Loading data

addpath('../../Tests');

load('CaseData1_0/tracos_in_time_ideal');
load('CaseData1_0/parameter');

%% Case 2.0 One primary and multiples

time = 0:dt:tmax;

% Plotting the trace
trace_1 = trace_p1_fst_prim_multiples_time(:, 1);

figure(1)
plot(time, trace_1)
grid

%% Getting the traning data (Matrix used in regression)

% In this ideal case, a filter with one coefficient is enough to recover
% the primary. Also, for this case, we can plot the function and regression
% line.

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

% We can algo plot the gain between the input sample and the sample to be
% predicted, we expect to a straight line of value 0.6

sample_gain = target./circshift(target', prediction_step)';

figure(4)
plot(time, sample_gain)
legend('Gain betwen sample to predicted and filter input sample')
xlim([0 1.5])
grid

% We can note some points out of the regression line, this points
% correspond to the primary that is not removed

%% We can also see the regression when the filter has length two

filter_one_len = 2;
prediction_step = 100;


[train_matrix, target] = trace_to_datatraining(trace_1, filter_one_len, prediction_step);

gain = inv(train_matrix*train_matrix')*train_matrix*target'

[mesh_x, mesh_y] = meshgrid(-1:0.1:1, -1:0.1:1);
regression_plan = [mesh_x(:), mesh_y(:)]*gain;
regression_plan = reshape(regression_plan, size(mesh_x));

% Regression for filter length 2
figure(4)
plot3(train_matrix(1, :), train_matrix(2, :), target,'.', train_matrix(1,:), train_matrix(2,:), gain'*train_matrix)
hold on
mesh(mesh_x, mesh_y, regression_plan)
view(30, 18);
grid

figure(5)
plot(time, target, time, target - gain'*train_matrix, '--')
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
