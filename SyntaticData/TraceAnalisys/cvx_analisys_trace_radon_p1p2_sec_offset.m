%% Loading data

clear all

addpath('../../Tests');
addpath('../../../../../../../cvx/')
load('../../SyntaticData/SimulatedDataGeneration/SynData_025/tracos_in_radon');
load('../../SyntaticData/SimulatedDataGeneration/SynData_025/parameter');

cvx_setup

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
plot(time, test_trace, 'r')
hold on
plot(time, reference_test_trace, 'b--')
legend('Primaries and multiples', 'Only primaries')
ylabel('Normalized Amplitude')
xlabel('\tau [s]')
xlim([0 time(1000)])
set(gca, 'FontSize', 12)
grid

%% Filterin by quadratic programing optmization

max_process_idx = 1:1e3;
n = max(max_process_idx);
test_trace = test_trace(max_process_idx);
reference_test_trace = reference_test_trace(max_process_idx);

filter_one_len = 1;
prediction_step = 96;

[reference_test_trace, avg, std_dev] = trace_normalization(reference_test_trace);
[train_matrix, avg, std_dev] = trace_normalization(test_trace);
[train_matrix, target] = trace_to_datatraining(train_matrix, filter_one_len, prediction_step);
target = target';
train_matrix = train_matrix';

hdiff = [1 -1];
Hdiff = convmtx(hdiff, n);
Hdiff(:, end) = [];
gdiff = [1 zeros(1, prediction_step-1) -1];
Gdiff = convmtx(gdiff, n);
Gdiff(:, end-(prediction_step-1):end) = [];

%% Filtering using CVX

cvx_begin
  variables x(n);
  minimize( norm(target - x.*train_matrix, 1) + 0.00*norm(x, 1) + 0.000*norm(x'*Hdiff, 2) + 0.00*norm(x, inf) );
  subject to
    -5 <= x <= 1
%      -5 <= x'*Gdiff <= 5
cvx_end


figure(2)
plot(target - x.*train_matrix)
hold on
plot([reference_test_trace(1:1e3-100)'],'g--')
legend('Recovered primaries', 'Primaries reference')
grid

%% Prediction step sensitivity analisys

prediction_steps = 86:106;

target_m = zeros(length(prediction_steps), length(target));
train_matrix_m = zeros(length(prediction_steps), length(train_matrix));
filtered_m = zeros(length(prediction_steps), length(train_matrix));
gain_m = zeros(length(prediction_steps), length(train_matrix));

for i = 1:length(prediction_steps)
  [train_matrix, target] = trace_to_datatraining(test_trace, filter_one_len, prediction_steps(i));
  target = target';
  train_matrix = train_matrix';
  target_m(i, :) = target;
  train_matrix_m(i, :) = train_matrix;

  cvx_begin
    variables x(n);
    minimize( norm(target - x.*train_matrix) );% + 0.001*norm(x'*Hdiff, 2) + 0.005*norm(x, inf) );
    subject to
      -2 <= x <= -0.1
      -0.2 <= x'*Gdiff <= 0.2
  cvx_end

  gain_m(i, :) = x;
  filtered_m(i, :) = target - x.*train_matrix;

end

%% Plotting sensitivity analisys results

figure(3)
plot(prediction_steps, sum(abs(filtered_m), 2), '.--')
xlabel('Prediction step')
ylabel('Trace MSE')
grid

figure(4)
plot(prediction_steps, var(filtered_m'), '.--')
xlabel('Prediction step')
ylabel('Trace Variance')
grid

figure(5)
plot(prediction_steps, mean(gain_m, 2), '.--')
xlabel('Prediction step')
ylabel('Gain Mean')
grid

figure(6)
plot(prediction_steps, var(gain_m'), '.--')
xlabel('Prediction step')
ylabel('Gain Variance')
grid

gmax = zeros(length(prediction_steps), 1);
gmin = zeros(length(prediction_steps), 1);
for i = 1:length(prediction_steps)
  gmax(i) = max(gain_m(i, :));
  gmin(i) = min(gain_m(i, :));
  gdiff = gmax-gmin;   
end

figure(7)
plot(prediction_steps, gmax, '.--')
xlabel('Prediction step')
ylabel('Gain max')
grid

figure(8)
plot(prediction_steps, gmin, '.--')
xlabel('Prediction step')
ylabel('Gain min')
grid

figure(9)
plot(prediction_steps, gdiff, '.--')
xlabel('Prediction step')
ylabel('Gain diff')
grid


%% if reversed gain has a very short period, erase it. Probrabli a spurios ripple
%% I think this can be add as a restrictions, verificar convolução que soma somente uma janela
% de elementos para filtrar
