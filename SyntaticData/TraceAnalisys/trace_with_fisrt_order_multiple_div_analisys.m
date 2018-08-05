%% Loading data

addpath('../../Tests');

load('CaseData1_0/tracos_in_time_ideal');
load('CaseData1_0/parameter');

%% Case 2.1 One primary and first order multiples - Shperic Divergence

time = 0:dt:tmax;

% Plotting the trace
trace_1 = trace_p1_fst_prim_multiples_div_time(:, 1);

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


%% Polinomial regression

% We can try use a polynomial instead of a linear regression

const = 0;
kernel_poly_matrix = (train_matrix'*train_matrix+const).^2;

% Second order polinomial
poly_features = [ones(1, size(train_matrix, 2)); train_matrix; train_matrix.^2; train_matrix.^3; train_matrix.^4];

gain_poly = inv(poly_features*poly_features')*poly_features*target'
poly_regression_gain = kernel_poly_matrix*inv(kernel_poly_matrix + 1*eye(size(kernel_poly_matrix)));

figure(5)
plot(train_matrix, target,'.', train_matrix, gain_poly'*poly_features)
legend('Function to be approximated', 'FIR with one delay aproximation')
grid

figure(6)
plot(time, target, time, target - gain_poly'*poly_features, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1.5])
grid

% The result is slightly improved, but it does overcome the limitations 
% of a feedfoward structure

%% Seen the regression problem by parts

idx = 501:601;

figure(7)
for i = 1:4
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


%% GMM
rng(10);

% Normalizing trace 
trace_1_norm = trace_normalization(trace_1);
trace_1_norm = trace_1_norm/max(trace_1_norm);
[train_matrix, target] = trace_to_datatraining(trace_1_norm, filter_one_len, prediction_step);

end_process = 801;
low_energy_idx = abs(target(1:end_process-1).^2) < 1e-3;

data_set = [train_matrix; target]';
data_set(end_process:end, :) = [];

init_gues = zeros(size(data_set,1), 1);
init_gues(50:150) = 1;
init_gues(151:end) = 2;
init_gues(low_energy_idx(1:end_process-1)) = 3;

gm = fitgmdist(data_set, 3, 'Start', init_gues, 'RegularizationValue', 1e-6);

figure(10)
h = ezcontour(@(x,y)pdf(gm,[x y]), [-1 1], [-1 1], 1e3);
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

figure(13)
plot(target(1:800))
hold on
plot(posterior(:, 3))
legend('Trace', 'Probability of low energy')
grid

%%
close all