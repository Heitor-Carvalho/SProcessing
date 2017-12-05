%% Loading data

addpath('../../Tests');

load('CaseData1_0/tracos_in_time');
load('CaseData1_0/parameter');

%% Case 2.1 - Anlisys of time trace - One primary and multiples

time = 0:dt:tmax;

% Plotting the trace
trace_1 = trace_p1_fst_prim_multiples(:, 1);

figure(1)
plot(time, trace_1)
grid

%% Getting the traning data (Matrix used in regression)

% In this ideal case, a filter with one coefficient is enough to recover
% the primary

filter_one_len = 1;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(trace_1, filter_one_len, prediction_step);

% With one coefficient, it's possible to visualize the function which the
%filter is trying to approximate, showed in Figure(2).

gain = inv(train_matrix*train_matrix')*train_matrix*target'

% Also in Figure (2), it can be seen that a straight line fits the data OK. We can
%also check the gain and compared with expected value of 0.6 (Water reflection coefficient)

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
%predicted, we expect to a straight line of value 0.6

sample_gain = target./circshift(target', prediction_step)';

figure(4)
plot(time, sample_gain)
legend('Gain betwen sample to predicted and filter input sample')
xlim([0 1.5])
grid

% Seen the Figures 4 we can see the gain change over time when two convolution
are made. Also, in Figure 3 we can see that the regression needs two map the sample
point in two different other points, which is impossible in feedfoward strucure.

% This cleary showns the limitations of such kinds of structure. However, we can still
use such structure by adding a new feature to the predictor.

% Fisrt, lets simple add time into the features a see what happens:

extend_train_matrix = [train_matrix; time];
extend_train_matrix = [train_matrix; train_matrix.*time];
extend_train_matrix = [train_matrix; train_matrix.^2];

extend_poly_features = [extend_train_matrix(1, :); extend_train_matrix(2, :); ...
                        extend_train_matrix(1,:).^2; extend_train_matrix(2, :).^2; ...
                        extend_train_matrix(1,:).^4; extend_train_matrix(2, :).^4; ...
                        extend_train_matrix(1,:).*extend_train_matrix(2,:)];

extend_poly_features = [extend_train_matrix(1, :); extend_train_matrix(2, :); extend_train_matrix(1,:).^2; extend_train_matrix(2, :).^2];

extend_poly_features = [extend_train_matrix(1, :); extend_train_matrix(2, :); extend_train_matrix(2, :).^2; ...
                        extend_train_matrix(1,:).*extend_train_matrix(2,:)];

extend_gain = inv(extend_train_matrix*extend_train_matrix')*extend_train_matrix*target'
extend_poly_gain = inv(extend_poly_features*extend_poly_features' + 1e-15*eye(size(extend_poly_features*extend_poly_features')))*extend_poly_features*target'
mse = mean((target - extend_poly_gain'*extend_poly_features).^2)
plot(time, target, time, extend_poly_gain'*extend_poly_features, '--')

mesh_y = 0:0.1:1;
mesh_x = -1:0.1:1;
[mesh_xx, mesh_yy] = meshgrid(mesh_x, mesh_y);

mesh_xy = [mesh_xx(:), mesh_yy(:)];
extend_poly_mesh = [mesh_xy(:, 1), mesh_xy(:, 2), mesh_xy(:, 1).^2, mesh_xy(:, 2).^2];
mesh_z = extend_poly_mesh*extend_poly_gain;

mesh_zz = reshape(mesh_z, size(mesh_xx))

mesh(mesh_xx, mesh_yy, mesh_zz)
hold on
plot3(extend_poly_features(1, :), extend_poly_features(2, :), extend_poly_gain'*extend_poly_features, 'o--')


figure(5)
plot(time, target, time, target - extend_gain'*extend_train_matrix, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

mesh_y = 0:0.1:tmax;
mesh_x = -1:0.1:1;
[mesh_xx, mesh_yy] = meshgrid(mesh_x, mesh_y);

mesh_xy = [mesh_xx(:), mesh_yy(:)];

mesh_z = mesh_xy*extend_gain;

mesh_zz = reshape(mesh_z, size(mesh_xx))

mesh(mesh_xx, mesh_yy, mesh_zz)
hold on
plot3(extend_train_matrix(1, :), extend_train_matrix(2, :), extend_gain'*extend_train_matrix, 'o--')
% By adding the time into the predictor we unfold the time solve and can still
use a feedfoward structure. However, now, or regression needs to map the intaire
trace!!. Which is a high non-linear functions

% Lets see if we can add a more sutable features and solve this problem:

% We need a feature that change with time, how about the gain!

sample_nan = isnan(sample_gain) | isinf(sample_gain)
sample_nan(1:100) = 1;
extend_train_matrix = [train_matrix(~sample_nan); sample_gain(~sample_nan)];
extend_target = target(~sample_nan);
extend_gain = inv(extend_train_matrix*extend_train_matrix')*extend_train_matrix*extend_target'

mesh_z = mesh_xy*extend_gain;

mesh_zz = reshape(mesh_z, size(mesh_xx))

plot(time(~sample_nan), extend_target, time(~sample_nan), extend_target - extend_gain'*extend_train_matrix, '--')

mesh(mesh_xx, mesh_yy, mesh_zz)
hold on
plot3(extend_train_matrix(1, :), extend_train_matrix(2, :), extend_gain'*extend_train_matrix,'o-')



TODO:
% In this case we noted some points out of the main line, this points due
% to the not perfect aligment of primary and multiples.

%% Now, let's add some noise and check this again!
trace_1_noisy = trace_1 + 0.02*randn(size(trace_1));

% First, lets look the noise trace
figure(3)
plot(time, trace_1_noisy)
grid

[train_matrix_noisy, target_noisy] = trace_to_datatraining(trace_1_noisy, filter_one_len, prediction_step);

gain_noisy = inv(train_matrix_noisy*train_matrix_noisy')*train_matrix_noisy*target_noisy'

figure(4)
plot(train_matrix_noisy, target_noisy,'.', train_matrix_noisy, train_matrix_noisy*gain_noisy)
legend('Function to be approximated', 'FIR with one delay aproximation')
grid

figure(5)
plot(time, target_noisy, time, target_noisy - train_matrix_noisy*gain, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

% Now, the noisy gain is not exactly 0.6 and we can see a cloud of points around
% the zero in the graph. This sugest that our linear regression in give more
% attention to the noise and those outliers point due to not perfect aligment.

%% Removing points with low energy

% We can try to eliminate the points with low energy, this way we can reduce
%the noise and the outlier points.

noisy_trace_target_energy = target_noisy.^2;
noisy_trace_matrix_energy = train_matrix.^2;

% Plotting energy time curve
figure(5)
plot(time, noisy_trace_energy)
grid

% Chosing a threshold and removing all points with energy below
% that we've got:
energy_threshold = 0.01;
removed_idx = mod(noisy_trace_energy < energy_threshold + noisy_trace_matrix_energy < energy_threshold, 2);

gain_noisy_less_points = inv(train_matrix_noisy(~removed_idx)*train_matrix_noisy(~removed_idx)')*train_matrix_noisy(~removed_idx)*target_noisy(~removed_idx)'

figure(6)
plot(train_matrix_noisy(~removed_idx), target_noisy(~removed_idx),'.', train_matrix_noisy(~removed_idx), train_matrix_noisy(~removed_idx)*gain_noisy)
legend('Function to be mapped points', 'FIR with one delay aproximation')
grid

figure(7)
plot(time, target_noisy, time, target_noisy - train_matrix_noisy*gain_noisy_less_points, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

% As can be seen, there was not much improvmente with this!

%% Robust linear regression

% We can try using a robust linear regression instead, this regression
%can better handle the outlier points. Although there's many parmeter
%that can be tunned, we will just run with the default value.

gain_noisy = robustfit(train_matrix_noisy, target_noisy);
gain_noisy = gain_noisy(2)
figure(8)
plot(train_matrix_noisy, target_noisy,'.', train_matrix_noisy, train_matrix_noisy*gain_noisy)
legend('Function to be approximated', 'FIR with one delay aproximation')
grid

figure(9)
plot(time, target_noisy, time, target_noisy - train_matrix_noisy*gain, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid



%%
close all
