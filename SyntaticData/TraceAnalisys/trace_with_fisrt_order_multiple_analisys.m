%% Loading data

addpath('../../Tests');

load('../../SyntaticData/SimulatedDataGeneration/SynData_025//tracos_in_time_ideal.mat');
load('../../SyntaticData/SimulatedDataGeneration/SynData_025//parameter');

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
figure(6)
plot(time, trace_1_noisy)
grid

filter_one_len = 1;
[train_matrix_noisy, target_noisy] = trace_to_datatraining(trace_1_noisy, filter_one_len, prediction_step);

gain_noisy = inv(train_matrix_noisy*train_matrix_noisy')*train_matrix_noisy*target_noisy'

figure(7)
plot(train_matrix_noisy, target_noisy,'.', train_matrix_noisy, train_matrix_noisy*gain_noisy)
legend('Function to be approximated', 'FIR with one delay aproximation')
grid

figure(8)
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
figure(9)
plot(time, noisy_trace_target_energy)
legend('Trace energy')
grid

% Chosing a threshold and removing all points with energy below,
% we've got:
energy_threshold = 0.01;
removed_idx = mod(noisy_trace_target_energy < energy_threshold, 2);

gain_noisy_less_points = inv(train_matrix_noisy(~removed_idx)*train_matrix_noisy(~removed_idx)')*train_matrix_noisy(~removed_idx)*target_noisy(~removed_idx)'

figure(10)
plot(train_matrix_noisy(~removed_idx), target_noisy(~removed_idx),'.', train_matrix_noisy(~removed_idx), train_matrix_noisy(~removed_idx)*gain_noisy)
legend('Function to be mapped points', 'FIR with one delay aproximation')
grid

figure(11)
plot(time, target_noisy, time, target_noisy - train_matrix_noisy*gain_noisy_less_points, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

% As can be seen, there was some improvment !!

%% GMM - Initing with fisrt primary
rng(1);

end_process = 801;

filter_one_len = 1;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(trace_1, filter_one_len, prediction_step);

data_set = [train_matrix; target]';
data_set(end_process:end, :) = [];

init_gues = 2*ones(size(data_set,1), 1);
init_gues(200:400) = 1;

gm = fitgmdist(data_set, 2, 'Start', init_gues, 'RegularizationValue', 1e-6);

figure(10)
h = ezcontour(@(x,y)pdf(gm,[x y]), [-1 1], [-1 1], 500);
hold on
plot(train_matrix(1:end_process), target(1:end_process),'.')
grid

posterior = gm.posterior(data_set);

figure(11)
plot(target(1:800))
hold on
plot(posterior(:, 1))
legend('Trace', 'Probability of been a primary')
grid

figure(12)
plot(target(1:800))
hold on
plot(posterior(:, 2))
legend('Trace', 'Probability of been a multiple')
grid

%% DataSet PCA

pca_scaling = -1:0.1:1;

[princomp] = pca(data_set);
p1 = kron(princomp(:,1), range)';
p2 = kron(princomp(:,2), range)';

figure(12)
plot(data_set(:, 1), data_set(:, 2))
hold on
plot(p1(:, 1), p1(:, 2))
plot(p2(:, 1), p2(:, 2))
legend('DataSet', 'Principal component 1', 'Principal component 2')
grid

%% Rotate principal components using promax criteria
% This allows to find not orthogonal components, which
% we hope results in more usefull components for our case

[rotate_promax, tpromax] = rotatefactors(data_set, 'Method', 'promax');
var(rotate_promax, 1)/min(var(rotate_promax, 1))

figure(13)
plot(data_set(:, 1), data_set(:, 2))
hold on
plot(rotate_promax(:, 1), rotate_promax(:, 2))
legend('DataSet', 'Rotate data')
grid

%% GMM - Initing clusters with component of more energy
rng(1);

% Normalizing trace
trace_1_norm = trace_normalization(trace_1);
trace_1_norm = trace_1_norm/max(trace_1_norm);
[train_matrix, target] = trace_to_datatraining(trace_1_norm, filter_one_len, prediction_step);

end_process = 801;
low_energy_idx = abs(target(1:end_process-1).^2) < 1e-3;

data_set = [train_matrix; target]';
data_set(end_process:end, :) = [];


idx_component_1 = rotate_promax(:, 1) < rotate_promax(:, 2);
idx_component_2 = rotate_promax(:, 2) <= rotate_promax(:, 1);

init_gues = 2*ones(size(data_set,1), 1);
init_gues(idx_component_1) = 1;

gm = fitgmdist(data_set, 2, 'Start', init_gues, 'RegularizationValue', 1e-6);

figure(10)
h = ezcontour(@(x,y)pdf(gm,[x y]), [-1 1], [-1 1], 500);
hold on
plot(train_matrix(1:end_process), target(1:end_process),'.')
grid

posterior = gm.posterior(data_set);

figure(11)
plot(target(1:800))
hold on
plot(posterior(:, 1))
legend('Trace', 'Probability of been a primary')
grid

figure(12)
plot(target(1:800))
hold on
plot(posterior(:, 2))
legend('Trace', 'Probability of been a multiple')
grid


%%
close all
