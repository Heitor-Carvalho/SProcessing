%% Loading data

addpath('../../Tests');
addpath('../../../IA353/Regularization/')
addpath('../../../IA353/NeuralNetwork/')
addpath('../../../IA353/ExtremeLearningMachine/')
addpath('../../../IA353/Regularization/')

load('../../SyntaticData/SimulatedDataGeneration/SynData_025//tracos_in_radon');
load('../../SyntaticData/SimulatedDataGeneration/SynData_025//parameter');

%% Case One primary and multiples - Zero offset

% Ploting filtered trace and reference trace

time = 0:dt:tmax;
trace_nb = 22;
attenuation_factor = 1;
samples_start = 1;

traces_matrix = radon_p1_fst_mul_div;
traces_matrix_prim = radon_p1_fst_prim_div;

% Nomalizing data
test_trace = trace_pre_processing(traces_matrix, trace_nb, samples_start, attenuation_factor);
reference_test_trace = trace_pre_processing(traces_matrix_prim, trace_nb, samples_start, attenuation_factor);

trace_nb = 22;
xlim_plot = 1000;

figure(1)
plot(time, test_trace,'--r')
hold on
plot(time, reference_test_trace, 'b')
ylabel('Normalized Amplitude')
legend('Primaries and multiples', 'Primary P1')
xlabel('\tau [s]')
xlim([0 time(1000)])
set(gca, 'FontSize', 12)
grid on
ylim([1 -1.5])

%% Trace autocorrelation

[trace_acc, lags] = xcorr(test_trace, 'coef');

figure(2)
plot(lags, trace_acc)
xlim([-1000 1000])
grid

%% FIR filter

filter_one_len = 2;
prediction_step = 100;

[train_matrix, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

gain = inv(train_matrix*train_matrix')*train_matrix*target'

figure(3)
plot(time, target, 'r--')
hold on
plot(time, target - gain'*train_matrix, 'b')
title('FIR - Filter')
legend('Primaries and multiples', 'Primary recovered')
ylabel('Normalized Amplitude')
xlabel('\tau [s]')
xlim([0 time(1000)])
set(gca, 'FontSize', 12)
grid

%% ELM 

prediction_step = 100;
filter_one_len = 5;   
mid_layer_sz = 56;
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


%% Plotting ELM - Results

figure(4)
plot(time, target, '--r')
hold on
plot(time, target - predicted_trace, 'b')
title('ELM')
legend('Primaries and multiples', 'Primary recovered')
ylabel('Normalized Amplitude')
xlabel('\tau [s]')
xlim([0 time(1000)])
set(gca, 'FontSize', 12)
grid

figure(5)
plot3(train_set(1, :), train_set(2, :), target, '--')
hold on
plot3(train_set(1, :), train_set(2, :), predicted_trace)
title('ELM - Regression space')
grid

%% ESN

prediction_step = 98;
filter_one_len = 5;   
mid_layer_sz = 56;
regularization = 1e-8;
initial_weigths_amp = 0.1;
spectral_radio = 0.1;

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

%% Plotting ESN - Results
figure(10)
plot(time, target, 'r--')
hold on
plot(time, target - predicted_trace,'b')
ylabel('Normalized Amplitude')
xlabel('\tau [s]')
xlim([0 time(1000)])
set(gca, 'FontSize', 12)
grid

figure(11)
plot3(train_set(1, :), train_set(2, :), target, '--')
hold on
plot3(train_set(1, :), train_set(2, :), predicted_trace)
title('ESN - Regression space')
grid

