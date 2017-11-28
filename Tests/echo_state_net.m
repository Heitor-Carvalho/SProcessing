addpath('../../IA353/NeuralNetwork/')
addpath('../../IA353/EchoStateNetworks/')
addpath('../../IA353/Regularization/')

% Loading traces matrix in Radon domain
data_set_name = './DataSets/tracos_radon_p2';
load(data_set_name)

debug_mode = 0; 

rng(1)
%% Test Script

% Trace pre-processing
trace_nb = 22;
attenuation_factor = 1;
samples_start = 1;
traces_matrix = radon_mult_offset150m_p2;
traces_matrix_prim = radon_prim_offset150m_p2;

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

prediction_step = 17;
filter_len = 10;   
mid_layer_sz = 10;
regularization = 0;
initial_weigths_amp = 0.9;
spectral_radio = 0.98;

% Variables used to generate plot figures
sweep_param = filter_len;
[file_name_ext, xlabel_txt] = net_analisys_text(1);

% Parameters lengths
prediction_step_params_len = length(prediction_step);
filter_params_len = length(filter_len);
mid_layer_params_len = length(mid_layer_sz);
regularization_params_len = length(regularization);

total_tests_nb = prediction_step_params_len*filter_params_len*mid_layer_params_len*regularization_params_len;

predicted_trace = zeros(length(trace_norm), total_tests_nb);
mse = zeros(total_tests_nb, 1);
mse_p = zeros(total_tests_nb, 1);

test_counter = 1;
for i = 1:length(prediction_step)
  for j = 1:length(regularization)
    for k = 1:length(filter_len)
      for l = 1:length(mid_layer_sz)

        % Neural network setup
        clear nn
        input_par.sz = [filter_len(k) mid_layer_sz(l)];
        input_par.range = initial_weigths_amp;
        % input_par.sparseness = 1;
        feedback_par.sz = [mid_layer_sz(l) mid_layer_sz(l)];
        feedback_par.range = initial_weigths_amp;
        feedback_par.alpha = spectral_radio;
%       feedback_par.sparseness = 1;
        out_sz = 1;
        nn.func = @tanh;
        nn.b = 0;

        [~, ~, W] = generate_echo_state_weigths(input_par, feedback_par);
        nn.v = W;
        nn = neuro_net_init(nn);

        % Preparing data based in parameters
        [train_set, target] = trace_to_datatraining(trace_norm, filter_len(k), prediction_step(i));

        % Calculating extreme learning machines values
        nn.w = calc_esn_weigths(train_set, target, regularization(j), nn);
        nn_{test_counter} = nn;

        % Neural network prediction
        predicted_trace(:, test_counter) = neural_net_echo_states(train_set, nn);
        mse(test_counter) = mean((predicted_trace(:, test_counter) - target').^2);
        mse_p(test_counter) = mean((target' - predicted_trace(:, test_counter) - trace_norm_prim).^2);

        test_counter = test_counter + 1;

      end
    end
  end
end

test_counter = test_counter - 1;

figure(3)
plot(0.2*trace_norm_prim,'linewidth',2)
xlim([0 250])
grid

figure(4)
subplot(2,1,1)
plot(0.2*(target' - predicted_trace(:, test_counter)),'linewidth',2)
hold on 
plot(0.2*(trace_norm), '--','linewidth',2)
plot(0.2*trace_norm_prim, '--g','linewidth',1)
legend('Traço Filtrado', 'Traço original')
xlim([0 250])
grid

subplot(2,1,2)
lagmax = 300;
acort_traco_filtrado = xcorr((target' - predicted_trace(:, test_counter)), 'biased');
plot(acort_traco_filtrado(length(trace_norm):length(trace_norm)+lagmax),'linewidth',2)
title('Autocorrelação do traço filtrado')
xlim([0 lagmax]); xlabel('lag'); grid;

parameter_error_gen_image
% Saving files
save(test_name, 'mse', 'mse_p', 'predicted_trace', 'nn_', 'trace_nb', 'samples_start', 'attenuation_factor', ...
                'mid_layer_sz', 'filter_len', 'prediction_step', 'regularization', 'initial_weigths_amp',  ...
                'data_set_name');





