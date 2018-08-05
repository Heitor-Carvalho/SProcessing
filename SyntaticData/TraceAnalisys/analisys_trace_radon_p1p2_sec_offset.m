%% Loading data

clear all

addpath('../../Tests');
addpath('../../../IA353/NeuralNetwork/')
addpath('../../../IA353/ExtremeLearningMachine/')
addpath('../../../IA353/EchoStateNetworks/')
addpath('../../../IA353/Regularization/')
addpath('../../Tests/GMM/')

load('CaseData1_0/tracos_in_radon');
load('CaseData1_0/parameter');

%% Case two primary and multiples - Zero offset

% Ploting filtered trace and reference trace
trace_nb = 22;
attenuation_factor = 1;
samples_start = 1;
time = 0:dt:tmax;

traces_matrix = radon_p1p2_sec_mul_div_offset;
traces_matrix_prim = radon_p1p2_primaries_div_offset;

% Nomalizing data
test_trace = trace_pre_processing(traces_matrix, trace_nb, samples_start, attenuation_factor);
reference_test_trace = trace_pre_processing(traces_matrix_prim, trace_nb, samples_start, attenuation_factor);

xlim_plot = 1000;

figure(1)
plot(time, test_trace, '--r')
hold on
plot(time, reference_test_trace, 'b')
legend('Primaries and multiples', 'Only primaries')
ylabel('Normalized Amplitude')
xlabel('\tau [s]')
xlim([0 time(1000)])
% set(gca, 'FontSize', 12)
grid

%% Trace autocorrelation

[trace_acc, lags] = xcorr(test_trace, 'coef');

figure(2)
plot(lags, trace_acc)
xlim([-1000 1000])
grid

%% FIR filter - One dimension

filter_one_len = 1;
prediction_step = 86;

[train_matrix, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

gain = inv(train_matrix*train_matrix')*train_matrix*target'

figure(3)
plot(target, 'r')
hold on
plot(target - gain'*train_matrix, 'b--')
title('FIR - Filter')
legend('Primaries and multiples', 'Primary recovered')
xlim([0 1000])
grid

figure(4)
plot(train_matrix, target,'--.', train_matrix, gain'*train_matrix)
legend('Function to be approximated', 'FIR with one delay aproximation')
axis([-2 2 -2 2])
grid

%% FIR filter - Two dimensions

filter_one_len = 2;
prediction_step = 90;

[train_matrix, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

gain = inv(train_matrix*train_matrix')*train_matrix*target'

[mesh_x, mesh_y] = meshgrid(-2:0.1:2, -2:0.1:2);
regression_plan = [mesh_x(:), mesh_y(:)]*gain;
regression_plan = reshape(regression_plan, size(mesh_x));

% Regression for filter length 2
figure(5)
plot3(train_matrix(1, :), train_matrix(2, :), target,'.', train_matrix(1,:), train_matrix(2,:), gain'*train_matrix)
hold on
mesh(mesh_x, mesh_y, regression_plan)
view(30, 18);
grid

figure(6)
plot(target)
hold on
plot(target - gain'*train_matrix, '--')
legend('Trace with primaries and multiples', 'Primary recovered')
xlim([0 1000])
grid

%% FIR filter

filter_one_len = 25;
prediction_step = 86;

[train_matrix, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

gain = inv(train_matrix*train_matrix')*train_matrix*target';

figure(7)
plot(target, 'r')
hold on
plot(target - gain'*train_matrix, 'b--')
title('FIR - Filter')
legend('Primaries and multiples', 'Primary recovered')
xlim([0 1000])
grid

%% FIR filter - Double Gate
filter_one_len = 25;
prediction_step = 86;

[train_matrix_1, target_1] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);
[train_matrix_2, target_2] = trace_to_datatraining(test_trace, filter_one_len, 2*prediction_step);

train_matrix = [train_matrix_1; train_matrix_2];
target = target_1 + target_2;

gain = inv(train_matrix*train_matrix')*train_matrix*target';

figure(8)
plot(target/2, 'r')
hold on
plot(target/2 - gain'*train_matrix/2, 'b--')
title('FIR - Double Gate')
legend('Primaries and multiples', 'Primary recovered')
xlim([0 1000])
grid

%% ELM - Adjusting prediction step 

filter_one_len = 20;   
mid_layer_sz = 56;
regularization = 0;
initial_weigths_amp = 0.1;

% Adjusting prediction step 
prediction_step = 80:110;

for i = 1:length(prediction_step)
  [train_set, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step(i));

  % Neural network setup
  clear nn
  in_sz = filter_one_len;
  out_sz = 1;
  nn.func = @tanh;
  nn.b = 0;

  % Calculating extreme learning machines values
  nn.v = initial_weigths_amp*(randn(in_sz+1, mid_layer_sz));
  nn = neuro_net_init(nn);
  nn.w = calc_elm_weigths(train_set, target, regularization, nn)';

  % Neural network prediction
  predicted_trace = neural_nete(train_set, nn);

  mse(i) = mean((predicted_trace - target).^2);
  mse_p(i) = mean((target - predicted_trace - reference_test_trace').^2);

end
 
% Plotting prediction error, and primary recovery 
figure(9)
plot(prediction_step, mse)
title('Prediction step error')
grid

figure(10)
plot(prediction_step, mse_p)
title('Rerovery vs reference primary error')
grid

%% ELM 
prediction_step = 83;
filter_one_len = 20;   
mid_layer_sz = 56;
regularization = 1e-8;
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
figure(11)
plot(target, '--')
hold on
plot(target - predicted_trace,'b')
plot(reference_test_trace, 'm')
title('ELM')
legend('Primaries and multiples', 'Primary recovered', 'Reference trace (Only primaries)', 'Location', 'southeast')
xlim([0 800])
grid

%% ELM Double Gate - Adjusting prediction step 

filter_one_len = 20;   
mid_layer_sz = 56;
regularization = 0;
initial_weigths_amp = 0.1;

% Adjusting prediction step 
prediction_step = 80:110;

for i = 1:length(prediction_step)
  [train_set_1, target_1] = trace_to_datatraining(test_trace, filter_one_len, prediction_step(i));
  [train_set_2, target_2] = trace_to_datatraining(test_trace, filter_one_len, 2*prediction_step(i));

  train_set = [train_set_1; train_set_2];
  target = target_1 + target_2;

  % Neural network setup
  clear nn
  in_sz = filter_one_len*2;
  out_sz = 1;
  nn.func = @tanh;
  nn.b = 0;

  % Calculating extreme learning machines values
  nn.v = initial_weigths_amp*(randn(in_sz+1, mid_layer_sz));
  nn = neuro_net_init(nn);
  nn.w = calc_elm_weigths(train_set, target, regularization, nn)';

  % Neural network prediction
  predicted_trace = neural_nete(train_set, nn);

  mse(i) = mean((predicted_trace/2 - target/2).^2);
  mse_p(i) = mean((target/2 - predicted_trace/2 - reference_test_trace').^2);

end
 
% Plotting prediction error, and primary recovery 
figure(91)
plot(prediction_step, mse)
title('Prediction step error')
grid

figure(101)
plot(prediction_step, mse_p)
title('Rerovery vs reference primary error')
grid

%% ELM - Double Gate
prediction_step = 84;
filter_one_len = 20;   
mid_layer_sz = 56;
regularization = 0;
initial_weigths_amp = 0.1;

[train_set_1, target_1] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);
[train_set_2, target_2] = trace_to_datatraining(test_trace, filter_one_len, 2*prediction_step);

train_set = [train_set_1; train_set_2];
target = target_1 + target_2;

% Neural network setup
clear nn
in_sz = filter_one_len*2;
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

figure(12)
plot(target/2, '--')
hold on
plot(target/2 - predicted_trace/2,'b')
plot(reference_test_trace, 'm')
title('Double gate - ELM')
legend('Primaries and multiples', 'Primary recovered', 'Reference trace (Only primaries)', 'Location', 'southeast')
xlim([0 800])
grid

%% ESN - Prediction step 

filter_one_len = 15;   
mid_layer_sz = 51;
regularization = 0;
initial_weigths_amp = 0.1;
spectral_radio = 0.1;

% Adjusting prediction step 
prediction_step = 70:100;

for i = 1:length(prediction_step)

  [train_set, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step(i));

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

  mse(i) = mean((predicted_trace - target).^2);
  mse_p(i) = mean((target - predicted_trace - reference_test_trace').^2);

end
 
% Plotting prediction error, and primary recovery 
figure(13)
plot(prediction_step, mse)
title('Prediction step error')
grid

figure(14)
plot(prediction_step, mse_p)
title('Rerovery vs reference primary error')
grid

%% ESN

prediction_step = 85;
filter_one_len = 10;   
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

% Plotting ESN - Results
figure(15)
plot(target, '--')
hold on
plot(target - predicted_trace,'b')
plot(reference_test_trace, 'm')
title('ESN')
legend('Primaries and multiples', 'Primary recovered', 'Reference trace (Only primaries)')
xlim([0 800])
grid

%% Studying the trainning set - Plotting data in regression space 

filter_one_len = 2;

% Highlithing the primaries
p1 = 50:150;
p2 = 330:390;

% 2D Space
filter_one_len = 2;
[train_set, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

figure(16)
plot(train_set(1, :), target)
hold on
plot(train_set(1, p1), target(p1))
plot(train_set(1, p2), target(p2), 'g')
legend('Primaries and Multiples', 'Primary P1', 'Primary P2')
axis([-1 1 -1 1])
grid

% 3D Space
filter_one_len = 3;
[train_set, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

figure(17)
plot3(train_set(1, :), train_set(2, :), target)
hold on
plot3(train_set(1, p1), train_set(2, p1), target(p1))
plot3(train_set(1, p2), train_set(2, p2), target(p2), 'g')
legend('Primaries and Multiples', 'Primary P1', 'Primary P2','Location','SouthOutside')
view(-70, 26);
grid

%% Fitting a GMM
rng(rand());

filter_one_len = 22;
prediction_step = 86;
[train_matrix, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

end_process = 801;

data_set = [train_matrix; target]';
data_set(end_process:end, :) = [];

init_gues = 2*ones(size(data_set,1), 1);

% Initializing the first cluster with the first primary information
init_gues(50:150) = 1;

gm = fitgmdist(data_set, 2, 'Start', init_gues, 'RegularizationValue', 1e-9, 'CovarianceType', 'Diagonal');

posterior = gm.posterior(data_set);

figure(18)
plot(target(1:800))
hold on
plot(posterior(:, 1))
legend('Trace', 'Probability of been a primary')
grid

figure(19)
plot(target(1:800))
hold on
plot(posterior(:, 2))
legend('Trace', 'Probability of been a multiple')
grid

%% Filtering the trace based on cluster classification - 50 % Decision threshold

tg = target(1:800);
tg(posterior(:, 1) < 0.5) = 0;
figure(20)
plot(tg)
legend('Trace with probability of primarie higher than 0.5')
grid

tg = target(1:800);
tg(posterior(:, 2) < 0.5) = 0;
figure(21)
plot(tg)
legend('Trace with probability of primarie higher than 0.5')
grid

%% ELM Space + PCA (To visualize)

% ELM transformation follwed by PCA

% Highlithing the primaries
p1 = 50:150;
p2 = 330:390;

filter_one_len = 15;   
mid_layer_sz = 51;
regularization = 0;
initial_weigths_amp = 0.1;
spectral_radio = 0.1;

% Adjusting prediction step 
prediction_step = 86;

[train_set, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

 % Neural network setup
clear nn
in_sz = filter_one_len;
out_sz = 1;
nn.func = @tanh;
nn.b = 0;

% Calculating extreme learning machines values
nn.v = initial_weigths_amp*(randn(in_sz+1, mid_layer_sz));
nn = neuro_net_init(nn);

[~, H] = calc_elm_weigths(train_set, target, regularization, nn);

[primcomp, projection] = pca(H(:, 2:end));

% ELM - 3D
figure(22)
plot3(projection(:, 1), projection(:, 2), target)
hold on
plot3(projection(p1, 1), projection(p1, 2), target(p1))
plot3(projection(p2, 1), projection(p2, 2), target(p2), 'g')
grid

% ELM - 2D
figure(23)
plot(projection(:, 1), target)
hold on
plot(projection(p1, 1), target(p1))
plot(projection(p2, 1), target(p2), 'g')
grid


%% Fitting a GMM - ELM Space

rng(rand())

end_process = 801;

filter_one_len = 25;
mid_layer_sz = 20;
regularization = 0;
initial_weigths_amp = 0.1;
prediction_step = 86;

% Neural network setup
clear nn
in_sz = filter_one_len;

out_sz = 1;
nn.func = @tanh;
nn.b = 0;

% Calculating extreme learning machines values
nn.v = initial_weigths_amp*(randn(in_sz+1, mid_layer_sz));
nn = neuro_net_init(nn);

[train_set, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_step);

[~, H] = calc_elm_weigths(train_set(:, 1:end_process-1), target(1:end_process-1), regularization, nn);
data_set = [H(:, 2:end), target(1:end_process-1)'];
data_set(end_process:end, :) = [];

%% GMM after ELM transformation

init_gues = 2*ones(size(data_set,1), 1);

% Initializing the first cluster with the first primary information
init_gues(50:150) = 1;

gm = fitgmdist(data_set, 2, 'Start', init_gues, 'RegularizationValue', 1e-9, 'CovarianceType', 'Diagonal');

posterior = gm.posterior(data_set);

figure(24)
plot(target(1:800))
hold on
plot(posterior(:, 1))
legend('Trace', 'Probability of been a primary')
grid

figure(25)
plot(target(1:800))
hold on
plot(posterior(:, 2))
legend('Trace', 'Probability of been a multiple')
grid

tg = target(1:800);
tg(posterior(:, 1) < 0.5) = 0;
figure(26)
plot(tg)
legend('Trace with probability of primarie higher than 0.5')
grid

tg = target(1:800);
tg(posterior(:, 2) < 0.5) = 0;
figure(27)
plot(tg)
legend('Trace with probability of primarie lower than 0.5')
grid

% At principle the clustering worked both before and after the ELM non-linear tranformation

%%
close all