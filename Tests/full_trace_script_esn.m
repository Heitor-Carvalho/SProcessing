% Loading traces matrix in Radon domain
data_case = 'Case3';
base_path = '../ExampleRafael/TestsMultiplasDifferentPosition/';
data_set_name = strcat(base_path, data_case, '/tracos_radon_p2');
data_set_name_time = strcat(base_path, data_case, '/tracos_tempo_p2');
data_set_parm_name = strcat(base_path, data_case, '/traco_parametros');
load(data_set_name)
load(data_set_name_time)
load(data_set_parm_name)

addpath('../ThirdParty/SeismicLab/codes/radon_transforms/')

rng(1)
%%

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;
traces_matrix = radon_mult_offset150m_p2;
traces_matrix_prim = radon_prim_offset150m_p2;

prediction_step = 15;
filter_len = 4;   
mid_layer_sz = 35;
regularization = 0;
initial_weigths_amp = 0.6;
spectral_radio = 0.9;

deconvolved_matrix = zeros(size(traces_matrix));
trace_nb = size(traces_matrix, 2);
mse_prediction = zeros(trace_nb, 1);
mse_reference_trace = zeros(trace_nb, 1);

for i=1:trace_nb
  
  % Nomalizing data
  [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, i, samples_start, attenuation_factor);
  trace_norm_prim = trace_pre_processing(traces_matrix_prim, i, samples_start, attenuation_factor);

  % Neural network setup
  clear nn
  input_par.sz = [filter_len mid_layer_sz];
  input_par.range = initial_weigths_amp;
  % input_par.sparseness = 1;
  feedback_par.sz = [mid_layer_sz mid_layer_sz];
  feedback_par.range = initial_weigths_amp;
  feedback_par.alpha = spectral_radio;
%   feedback_par.sparseness = 1;
  i
  out_sz = 1;
  nn.func = @tanh;
  nn.b = 0;

  [~, ~, W] = generate_echo_state_weigths(input_par, feedback_par);
  nn.v = W;
  nn = neuro_net_init(nn);

  % Preparing data based in parameters
  [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);

  % Calculating extreme learning machines values
  nn.w = calc_esn_weigths(train_set, target, regularization, nn);

  % Apply network to all traces
  deconvolved_matrix(:, i) = target - neural_net_echo_states(train_set, nn);
  mse_prediction(i) = mean(deconvolved_matrix(:, i).^2);
  mse_reference_trace(i) = mean((deconvolved_matrix(:, i) - trace_norm_prim).^2);
  deconvolved_matrix(:, i) = deconvolved_matrix(:, i)*max_amp*std_dev + avg;

end

primaries = forward_radon_freq(traces_matrix_prim,dt,h_fo150,q,1,flow,fhigh);
prim_est = forward_radon_freq(deconvolved_matrix,dt,h_fo150,q,1,flow,fhigh);
multiples = forward_radon_freq(traces_matrix,dt,h_fo150,q,1,flow,fhigh);

% Plotting processed traces image
figure(1)
imagesc(primaries(1:200, 1:200), [-1 1]*1e-3)
title('Traço somente primárias (após transformada Radon)')
grid
saveas(gcf, sprintf('traco_primarias_%s.png', lower(data_case)));

figure(2)
imagesc(multiples(1:200, 1:200), [-1 1]*1e-3)
title('Traço com múltiplas (inversa Radon)')
grid
saveas(gcf, sprintf('traco_multiplas_%s.png', lower(data_case)));

figure(3)
imagesc(prim_est(1:200, 1:200), [-1 1]*1e-3)
title('Traço filtrado (após a remoção das múltiplas')
grid
saveas(gcf, sprintf('traco_filtrado_%s.png', lower(data_case)));

figure(4)
plot(primaries(:, 1))
title('Traço 1 no tempo contendo somente as duas primárias')
xlim([0 250])
grid
saveas(gcf, sprintf('traco1_tempo_%s.png', lower(data_case)));

save(lower(data_case), 'primaries', 'prim_est', 'multiples', 'nn', 'data_case')