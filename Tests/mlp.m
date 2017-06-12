addpath('../../IA353/NeuralNetwork/')
addpath('../../IA353/TrainingMethods')
addpath('../../IA353/BackPropagation')
addpath('../../IA353/LineSearchs')

% Loading traces matrix in Radon domain
data_set_name = './DataSets/tracos_radon_p1';
load(data_set_name)

debug_mode = 0; 

%% Test Script

% Trace pre-processing
trace_nb = 22;
attenuation_factor = 1;
samples_start = 1;
traces_matrix = radon_mult_offset150m_p1;
traces_matrix_prim = radon_prim_offset150m_p1;

% Nomalizing data
trace_norm = trace_pre_processing(traces_matrix, trace_nb, samples_start, attenuation_factor);
trace_norm_prim = trace_pre_processing(traces_matrix_prim, trace_nb, samples_start, attenuation_factor);

% Trace autocorrlation
if(debug_mode)
  [trace_acc, lags] = xcorr(trace_norm, 'coef');
  figure(1)
  plot(lags, trace_acc,'o-')
  xlim([-100 100])
  grid
  figure(2)
  plot(trace_norm)
  grid
end

% Test data name
test_name = 'TT';

sample_to_predict = 30;
filter_len = 10;   
mid_layer_sz = 60;
regularization = 0;
initial_weigths_amp = 1;

% Parameters lengths
sample_to_predict_params_len = length(sample_to_predict);
filter_params_len = length(filter_len);
mid_layer_params_len = length(mid_layer_sz);
regularization_params_len = length(regularization);

total_tests_nb = sample_to_predict_params_len*filter_params_len*mid_layer_params_len*regularization_params_len;

predicted_trace = zeros(length(trace_norm), total_tests_nb);
mse = zeros(total_tests_nb, 1);
mse_p = zeros(total_tests_nb, 1);

test_counter = 1;

train_par.max_it = 50;
train_par.max_error = 1e-6;

for i = 1:length(sample_to_predict)
  for j = 1:length(regularization)
    for k = 1:length(filter_len)
      for l = 1:length(mid_layer_sz)

        % Neural network setup
        clear nn
        in_sz = filter_len(k);
        out_sz = 1;
        nn.func = @tanh;
        nn.diff = @(x) 1 - tanh(x).^2;
        nn.b = 0;

        nn.v = initial_weigths_amp*(rand(in_sz+1, mid_layer_sz(l)));
        nn.w = initial_weigths_amp*rand(1, mid_layer_sz+1);
        nn = neuro_net_init(nn);

        % Preparing data based in parameters
        [train_set, target] = trace_to_datatraining(trace_norm, filter_len(k), sample_to_predict(i)-filter_len(k));

        % Calculating extreme learning machines values
        [nnt] = batch_cg_bfgs_training(train_set, target, nn, train_par, 1e-3)';
        nn_{test_counter} = nn;

        % Neural network prediction
        predicted_trace(:, test_counter) = neural_nete(train_set, nn);
        mse(test_counter) = mean((predicted_trace(:, test_counter) - target').^2);
        mse_p(test_counter) = mean((target' - predicted_trace(:, test_counter) - trace_norm_prim).^2);

        test_counter = test_counter + 1;

      end
    end
  end
end


% Saving files
save(test_name, 'mse', 'mse_p', 'predicted_trace', 'nn_', 'trace_nb', 'samples_start', 'attenuation_factor', ...
                'mid_layer_sz', 'filter_len', 'sample_to_predict', 'regularization', 'initial_weigths_amp',  ...
                'data_set_name');





