%% Loading data

addpath('../../Tests');
addpath('../../Tests/RLS')
addpath('../../../IA353/NeuralNetwork/')
addpath('../../../IA353/EchoStateNetworks/')
addpath('../../../IA353/Regularization/')
addpath('../../Tests/GMM')

load('CaseData1_0/tracos_in_radon');
load('CaseData1_0/parameter');

%% Case two primary and multiples - Zero offset

% Ploting filtered trace and reference trace

trace_nb = 100;
attenuation_factor = 1;
samples_start = 1;

traces_matrix = radon_p1p2_sec_mul_div;
traces_matrix_prim = radon_p1p2_primaries_div;

% Nomalizing data
test_trace = trace_pre_processing(traces_matrix, trace_nb, samples_start, attenuation_factor);
reference_test_trace = trace_pre_processing(traces_matrix_prim, trace_nb, samples_start, attenuation_factor);

trace_nb = 22;
xlim_plot = 1000;

figure(1)
plot(test_trace)
hold on
plot(reference_test_trace)
legend('Primaries and multiples', 'Primary P1 and P2')
xlim([0 1000])
grid

%% Trace autocorrelation

[trace_acc, lags] = xcorr(test_trace, 'coef');

figure(2)
plot(lags, trace_acc)
xlim([-1000 1000])
grid


%% FIR filter

filter_one_len = 5;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

gain = inv(train_matrix*train_matrix')*train_matrix*target';

figure(3)
plot(target, '--')
hold on
plot(target - gain'*train_matrix,'b')
plot(reference_test_trace, 'm')
title('FIR - Filter')
legend('Primaries and multiples', 'Primary recovered', 'Reference trace (Only primaries)')
xlim([0 800])
grid

%% ELM 

prediction_step = 99;
filter_one_len = 2;   
mid_layer_sz = 35;
regularization = 0;
initial_weigths_amp = 0.1;

[train_set, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

% Neural network setup
clear nn
in_sz = filter_one_len;
out_sz = 1;
nn.func = @tanh;
nn.b = 0;

% Calculating extreme learning machines values
nn.v = initial_weigths_amp*(rand(in_sz+1, mid_layer_sz));
nn = neuro_net_init(nn);
nn.w = calc_elm_weigths(train_set, target, regularization, nn)';

% Neural network prediction
predicted_trace = neural_nete(train_set, nn);

mse = mean((predicted_trace - target).^2);
mse_p = mean((target - predicted_trace - reference_test_trace').^2);

% Plotting ELM - Results

figure(4)
plot(target, '--')
hold on
plot(target - predicted_trace,'b')
plot(reference_test_trace, 'm')
title('ELM')
legend('Primaries and multiples', 'Primary recovered', 'Reference trace (Only primaries)')
xlim([0 800])
grid

figure(5)
plot3(train_set(1, :), train_set(2, :), target, '--')
hold on
plot3(train_set(1, :), train_set(2, :), predicted_trace)
title('ELM - Regression space')
grid

%%  FIR adaptative - RLS

filter_one_len = 2;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

% Initiating RLS with OLS solution and residual variance
gain = inv(train_matrix*train_matrix')*train_matrix*target';
residual_var = var(target - gain'*train_matrix);

init_cov_matrix = residual_var;
w0 = gain;
lambda = 0.98;

[y_est, w, ~] = rls(train_matrix, target, init_cov_matrix, w0, lambda);

figure(6)
plot(target, '--')
hold on
plot(target-y_est, 'b')
plot(reference_test_trace, 'm')
title('FIR adptative - RLS')
legend('Primaries and multiples', 'Primary recovered', 'Reference trace (Only primaries)')
xlim([0 1000])
grid

%% ESN 

prediction_step = 95;
filter_one_len = 2;   
mid_layer_sz = 31;
regularization = 1e-3;
initial_weigths_amp = 0.5;
spectral_radio = 0.98;

[train_set, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

% Neural network setup
clear nn
input_par.sz = [filter_one_len mid_layer_sz];
input_par.range = initial_weigths_amp;
feedback_par.sz = [mid_layer_sz mid_layer_sz];
feedback_par.range = initial_weigths_amp;
feedback_par.alpha = spectral_radio;
out_sz = 1;
nn.func = @tanh;
nn.b = 0;

[~, ~, W] = generate_echo_state_weigths(input_par, feedback_par);
nn.v = W;
nn = neuro_net_init(nn);

% Calculating extreme learning machines values
nn.w = calc_esn_weigths(train_set, target, regularization, nn);

predicted_trace = neural_net_echo_states(train_set, nn);

mse = mean((predicted_trace - target).^2);
mse_p = mean((target - predicted_trace - reference_test_trace').^2);

% Plotting ESN - Results
figure(9)
plot(target, '--')
hold on
plot(target - predicted_trace,'b')
plot(reference_test_trace, 'm')
title('ESN')
legend('Primaries and multiples', 'Primary recovered', 'Reference trace (Only primaries)')
xlim([0 800])
grid

figure(10)
plot3(train_set(1, :), train_set(2, :), target, '--')
hold on
plot3(train_set(1, :), train_set(2, :), predicted_trace)
title('ESN - Regression space')
grid

%% GMM
rng(10);

prediction_step = 100;
filter_one_len = 6;
[train_matrix, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);


number_of_components = 2;

end_process = 801;
low_energy_idx = abs(target(1:end_process-1).^2) < 1e-4;

data_set = [train_matrix; target]';
data_set(end_process:end, :) = [];

idxp = 108:180;
idxp2 = 338:379;

mix_prob = [0.3 0.7];
mix_cov = zeros(size(data_set, 2), size(data_set, 2), number_of_components);
mix_cov(:,:,1) = cov(data_set(idxp, :));
mix_cov(:,:,2) = cov(data_set(setxor(idxp, 1:size(data_set, 1)), :));
mix_mean = zeros(1, size(data_set, 2), number_of_components);
mix_mean(:,:,1) = mean(data_set(idxp, :));
mix_mean(:,:,2) = mean(data_set(setxor(idxp, 1:size(data_set, 1)), :));
v = 1;

max_it = 1e5;
% [mix_prob, mix_cov, mix_mean] = gmm_em(data_set, mix_prob, mix_cov, mix_mean, 1e-8, max_it);
[mix_prob, mix_cov, mix_mean] = tstudentmm_em(data_set, mix_prob, mix_cov, mix_mean, v, 1e-8, max_it);
%%

  for i = 1:number_of_components
    mix_cov(:, :, i)  = mix_cov(:, :, i) + 1e-11*eye(size(mix_cov(:, :, i)));
  end 

[posterior_prob, posterior] = tstudentmm_posterior(data_set, mix_prob, mix_cov, mix_mean, v);
% [posterior_prob, posterior] = gmm_posterior(data_set, mix_prob, mix_cov, mix_mean);


figure(19)
plot(target(1:800))
hold on
plot(posterior_prob(:, 1))
legend('Trace', 'Probability of been a primary')
grid

figure(20)
plot(target(1:800))
hold on
plot(posterior_prob(:, 2))
legend('Trace', 'Probability of been a multiple')
grid
%%
figure(23)
tgp = target(1:800);
tgp(posterior_prob(:, 1) < 0.5) = 0;
plot(tgp)
grid

%%
figure(24)
tgm = target(1:800);
tgm(posterior_prob(:, 2) < 0.5) = 0;
plot(tgm)
grid
%%

Wp = diag(posterior_prob(:, 1));
% Wp = diag(posterior_prob(:, 1) > 0.5);
WWp = data_set(:, 1:end-1)'*Wp*data_set(:, 1:end-1);

W = diag(posterior_prob(:, 2));
% W = diag(posterior_prob(:, 2) > 0.5);
WW = data_set(:, 1:end-1)'*W*data_set(:, 1:end-1);

% reg_p = 0;
reg_m = 0;
gain_prim = inv(WWp + reg_p*eye(size(WWp)))*data_set(:, 1:end-1)'*Wp*target(1:end_process-1)';
gain_multiple = pinv(WW + reg_m*eye(size(WW)))*data_set(:, 1:end-1)'*W*target(1:end_process-1)';

%%
figure(25)
plot3(data_set(:, 1), data_set(:, 2), target(1:end_process-1))
hold on
plot3(data_set(idxp, 1), data_set(idxp, 2), target(idxp), '--r', 'LineWidth', 2.5)
plot3(data_set(idxp2, 1), data_set(idxp2, 2), target(idxp2), '--g', 'LineWidth', 2.5)
grid on
%%
figure(21)
plot(target(1:800))
hold on
plot(gain_multiple'*data_set(:, 1:end-1)', '--r')
ylim([-2 1])
grid
%%
figure(22)
plot(target(1:800))
hold on
% plot(target(1:800) - gain_multiple'*data_set(:, 1:end-1)', '--m')
plot(gain_prim'*data_set(:, 1:end-1)', 'r--')
grid
% plot(target(1:800))
% hold on
% plot(posterior(:, 3))
% legend('Trace', 'Probability of been a low energy')
% grid



%%
gm = fitgmdist(data_set, 3, 'Start', init_gues, 'RegularizationValue', 1e-6);

figure(11)
h = ezcontour(@(x,y)pdf(gm,[x y]), [-1 1], [-1 1], 1e3);
hold on
plot(train_matrix(1:end_process), target(1:end_process),'.')
grid

posterior = gm.posterior(data_set);

figure(12)
plot(target(1:800))
hold on
plot(posterior(:, 1))
plot(reference_test_trace(1:800))
legend('Trace', 'Probability of been a primary', 'Reference primary')
grid

figure(13)
plot(target(1:800))
hold on
plot(posterior(:, 2))
plot(target(1:800))
hold on
plot(posterior(:, 1))
plot(reference_test_trace(1:800))
legend('Trace', 'Probability of been a primary', 'Reference primary')
grid
legend('Trace', 'Probability of been a multiple')
grid

figure(14)
plot(target(1:800))
hold on
plot(posterior(:, 3))
legend('Trace', 'Probability of low energy')
grid

%%
close all