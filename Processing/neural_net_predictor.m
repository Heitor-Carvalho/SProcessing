clear all
% Loading test trace
test_trace = load('test_trace.mat');

% Regulatization
regularization = 0;

% Neural network as a curve predictor
samples_start = 21;
filter_len = 15;

% Middle layer size
min_sz = 1;
max_sz = 100;

mse_error = zeros(length(filter_len), max_sz-min_sz+1);

for i = 1:length(filter_len)
  predictor_len = samples_start-filter_len(i);
  mse_error(i, :) = test_midsize_neural_net(test_trace, min_sz, max_sz, filter_len, predictor_len, regularization);
end



