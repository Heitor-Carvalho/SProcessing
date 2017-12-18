%% Loading data

addpath('../../Tests');
addpath('../../Tests/RLS')
addpath('../../../IA353/NeuralNetwork/')
addpath('../../../IA353/EchoStateNetworks/')
addpath('../../../IA353/Regularization/')

load('CaseData1_0/tracos_in_radon');
load('CaseData1_0/parameter');

%% Case two primary and multiples - Zero offset

% Ploting filtered trace and reference trace

trace_nb = 22;
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
filter_one_len = 12;   
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

%%
close all