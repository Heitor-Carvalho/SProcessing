%% Loading data

addpath('../../Tests');

load('CaseData1_0/tracos_in_time_ideal');
load('CaseData1_0/parameter');

%% Case 6.0 One primary and second order multiples

time = 0:dt:tmax;

% Plotting the trace
trace_1 = trace_p2_sec_prim_multiples_time(:, 1);

figure(1)
plot(trace_1)
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
plot(train_matrix, target,'--.', train_matrix, train_matrix*gain)
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
for i = 1:10
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
rng(1);
end_process = 801;

filter_one_len = 1;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(trace_1, filter_one_len, prediction_step);

data_set = [train_matrix; target]';
data_set(end_process:end, :) = [];

init_gues = 2*ones(size(data_set,1), 1);
init_gues(300:400) = 1;
init_gues(setxor(300:400, 1:length(init_gues))) = 2;

gm = fitgmdist(data_set, 2, 'Start', init_gues, 'RegularizationValue', 1e-7);

figure(12)
h = ezcontour(@(x,y)pdf(gm,[x y]),[-1 1],[-1 1], 800);
hold on
plot(train_matrix(1:end_process), target(1:end_process),'.')
grid

posterior = gm.posterior(data_set);

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

%%

tg = target(1:800);
tg(posterior(:, 1) < 0.5) = 0;
figure(15)
plot(tg)
legend('Trace with probability of primarie higher than 0.5')
grid

tg = target(1:800);
tg(posterior(:, 2) < 0.5) = 0;
figure(16)
plot(tg)
legend('Trace with probability of primarie higher than 0.5')
grid

%% TMM

number_of_components = 2;
idxp = 300:400;
mix_prob = [0.5 0.5];
mix_cov = zeros(size(data_set, 2), size(data_set, 2), number_of_components);
mix_cov(:,:,1) = cov(data_set(idxp, :));
mix_cov(:,:,2) = cov(data_set(setxor(idxp, 1:size(data_set, 1)), :));
mix_mean = zeros(1, size(data_set, 2), number_of_components);
mix_mean(:,:,1) = mean(data_set(300:400, :));
mix_mean(:,:,2) = mean(data_set(setxor(300:400, 1:size(data_set, 1))));
v = 1e3;

max_it = 5e4;
[mix_prob, mix_cov, mix_mean] = tstudentmm_em(data_set, mix_prob, mix_cov, mix_mean, v, 1e-9, max_it);

[posterior_prob, posterior] = tstudentmm_posterior(data_set, mix_prob, mix_cov, mix_mean, v);
post_func = @(x, y) tstudentmm_pdf([x, y], mix_prob, mix_cov, mix_mean, v);

figure(17)
h = ezcontour(post_func,[-1 1],[-1 1], 800);
hold on
plot(train_matrix(1:end_process), target(1:end_process),'.')
grid

figure(18)
plot(target(1:800))
hold on
plot(posterior_prob(:, 1))
legend('Trace', 'Probability of been a primary')
grid

figure(19)
plot(target(1:800))
hold on
plot(posterior_prob(:, 2))
legend('Trace', 'Probability of been a multiple')
grid

%%

tg = target(1:800);
tg(posterior(:, 1) < 0.5) = 0;
figure(15)
plot(tg)
legend('Trace with probability of primarie higher than 0.5')
grid

tg = target(1:800);
tg(posterior(:, 2) < 0.5) = 0;
figure(16)
plot(tg)
legend('Trace with probability of primarie higher than 0.5')
grid


%%
close all