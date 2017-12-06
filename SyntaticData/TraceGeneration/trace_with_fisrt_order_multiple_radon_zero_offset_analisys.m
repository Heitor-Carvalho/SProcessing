%% Loading data

addpath('../../Tests');

load('CaseData1_0/tracos_in_radon');
load('CaseData1_0/parameter');

%% Case 2.4 - Anlisys of time trace - One primary and first order multiples - Shperic Divergence

time = 0:dt:tmax;

% Plotting the trace
trace_1 = radon_p1_fst_mul_div(:, 22);

figure(1)
plot(time, trace_1)
grid

%% Getting the traning data (Matrix used in regression)

% In this case, the function to be aproximated by the filter
% is no longer a line. That's why the a linear filter it's not
% enough to recover the primary

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

sample_gain = target./circshift(target', prediction_step)';

figure(4)
plot(time(500+1:end), sample_gain(500+1:end), '-')
legend('Gain betwen sample to predicted and filter input sample')
xlim([0 1.5])
grid

% We can see, that in this case, the sample gain changes over time, this
% showns that or line aproximation must change with time.
% Also, by looking at function to be aproximated we can note that or 
% function as two points in y-axis for the same x-axis points. This results
% shows that a single feedfoward structures in uncapable of removing the
% primaryes complete

%% Linear regression with two coefficients

filter_one_len = 2;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(trace_1, filter_one_len, prediction_step);

gain = inv(train_matrix*train_matrix')*train_matrix*target'

[mesh_x, mesh_y] = meshgrid(-4:0.2:4, -4:0.2:4);
mesh_x = mesh_x*1e-5;
mesh_y = mesh_y*1e-5;
regression_plan = [mesh_x(:), mesh_y(:)]*gain;
regression_plan = reshape(regression_plan, size(mesh_x));

figure(2)
plot3(train_matrix(1, :), train_matrix(2, :), target,'.', train_matrix(1, :), train_matrix(2, :), gain'*train_matrix)
grid

figure(3)
plot(time, target, time, target - gain'*train_matrix, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

figure(4)
plot3(train_matrix(1, :), train_matrix(2, :), target,'.', train_matrix(1,:), train_matrix(2,:), gain'*train_matrix)
hold on
mesh(mesh_x, mesh_y, regression_plan)
view(30, 18);
grid



%% Polinomial regression

% We can try use a polynomial instead of a linear regression

const = 0;
kernel_poly_matrix = (train_matrix'*train_matrix+const).^2;

% Second order polinomial
poly_features = [ones(1, size(train_matrix, 2)); train_matrix; train_matrix.^2; train_matrix.^3; train_matrix.^4];

gain_poly = inv(poly_features*poly_features')*poly_features*target'
poly_regression_gain = kernel_poly_matrix*inv(kernel_poly_matrix + 1*eye(size(kernel_poly_matrix)));

figure(6)
plot(train_matrix, target,'.', train_matrix, gain_poly'*poly_features)
legend('Function to be approximated', 'FIR with one delay aproximation')
grid

figure(7)
plot(time, target, time, target - gain_poly'*poly_features, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

% The result is slightly improved, but it does overcome the limitations 
% of a feedfoward structure

%% Seen the regression problem by parts

idx = 501:601;

figure(8)
for i = 1
  plot(train_matrix(idx+(i-1)*100), target(idx+(i-1)*100),'.-')
  hold on
end
grid

% We can see that in this case, instead of one single line we have 
% many lines. One for each combination of reflextions of n-th order (see
% Vershuur, page 56)
% This suggest to aproach to the problem:
% 1 - Use a structure we time variyng parameters or recorrent structure
% 2 - Use a many feedfoward structure ajusted simultaneously


%%
close all