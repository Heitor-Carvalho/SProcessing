function [train_set, target] = trace_to_datatraining(trace, filter_len, prediction_len)
% [train_set, target] = trace_to_datatraining(trace, filter_len, prediction_len)
% - Generate the training set and target used to train the neural network.
% Inputs:
%  trace          - Vector with the seismic trace
%  filter_len     - Predictor filter length
%  prediction_len - Prediction step
% Outputs:
%  train_set      - matrix with one training pattern per line
%  target         - reference to each pattern
  
  training_sample = length(trace)-filter_len-prediction_len+1;

  target = zeros(1, training_sample);
  target(:) = trace(filter_len+prediction_len:end)';

  train_set = zeros(filter_len, training_sample);
  for i = 1:length(trace)-filter_len-prediction_len+1
    train_set(:, i) = trace(i:i+filter_len-1);     
  end

end