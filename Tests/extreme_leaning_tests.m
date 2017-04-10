addpath('../../IA353/ExtremeLearningMachine/')
addpath('../../IA353/NeuralNetwork/')

% Loading traces matrix in Radon domain
data_set_name = '../SyntheticTrace/trace_example_radon_domain';
load(data_set_name)

debug_mode = 0; 

%% Test 1 
% - Synthetic trace with only one primaries and
% and multiples. 
% - Multiple removal in Radon domain
% - Evaluation of the multiple removal variyng:
%   -> Number of hidden neuron layers
%   -> Samples to be predicted (prediction length + filter_len)
%   -> Filter length
%   -> Prediction length

% Trace pre-processing
trace_nb = 22;
attenuation_factor = 0.2;
samples_start = 19;

% Removing firt zeros samples and nomalizing data
trace_norm = trace_pre_processing(radon_mult_fo150, trace_nb, samples_start, attenuation_factor);

% Trace autocorrlation
if(debug_mode)
  [trace_acc, lags] = xcorr(trace_norm, 'coef');
  plot(lags, trace_acc,'o-')
  xlim([-100 100])
  grid
end

sample_to_predict = 38;
filter_len = [1 2 4 6 8 10 12 14 16 18];
mid_layer_sz = 1:2:90;
regularization = 0;

predicted_trace = zeros(length(trace_norm)-sample_to_predict+1, length(mid_layer_sz), length(filter_len));
mse = zeros(1, length(mid_layer_sz), length(filter_len));

test_counter = 0;
for i = 1:length(filter_len)
  for j = 1:length(mid_layer_sz)

    % Neural network setup
    clear nn
    in_sz = filter_len(i);
    out_sz = 1;
    nn.func = @tanh;
    nn.b = 0;
    nn.v = 1*rand(in_sz+1, mid_layer_sz(j));
    nn = neuro_net_init(nn);

    % Preparing data based in parameters
    [train_set, target] = trace_to_datatraining(trace_norm, filter_len(i), sample_to_predict-filter_len(i));
    
    % Calculating extreme learning machines values
    nn.w = calc_elm_weigths(train_set, target, regularization, nn)';
    nn_{test_counter+1} = nn;
    
    % Neural network prediction
    predicted_trace(:, j, i) = neural_nete(train_set, nn);
    mse(:, j, i) = mean((predicted_trace(:, j, i) - target').^2);

    test_counter = test_counter + 1;
    
  end
end

% Test data name
test_name = sprintf('trace_%d_predict_sample_%d', trace_nb, sample_to_predict);

% Saving files
save(test_name, 'mse', 'predicted_trace', 'nn_'     , 'mid_layer_sz', 'filter_len'         , ...
                'sample_to_predict'     , 'trace_nb', 'data_set_name', 'attenuation_factor', ...
                'samples_start');





