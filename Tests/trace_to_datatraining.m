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
  

  train_set = toeplitz(zeros(filter_len, 1),[zeros(1, prediction_len) trace(1:end - prediction_len)']);
  target = trace';

end