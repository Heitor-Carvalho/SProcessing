% Loading traces matrix in Radon domain
data_case = 'Case5';
base_path = '../ExampleRafael/TestsMultiplasDifferentPosition/';
data_set_name = strcat(base_path, data_case, '/tracos_radon_p2');
data_set_name_time = strcat(base_path, data_case, '/tracos_tempo_p2');
data_set_parm_name = strcat(base_path, data_case, '/traco_parametros');
load(data_set_name)
load(data_set_name_time)
load(data_set_parm_name)

addpath('../ThirdParty/SeismicLab/codes/radon_transforms/')


%%

% Plotting parameters
wiggle_th = 120;

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;
traces_matrix = real_data_muted;
traces_matrix_prim = zeros(size(real_data_muted));

% Using cursor information
prediction_step = 0;
filter_len = 6;   
mid_layer_sz = 15;
regularization = 0;
initial_weigths_amp = 0.5;

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
  in_sz = filter_len;
  out_sz = 1;
  nn.func = @tanh;
  nn.b = 0;

  nn.v = initial_weigths_amp*(rand(in_sz+1, mid_layer_sz));
  nn = neuro_net_init(nn);

  % Preparing data based in parameters
  % Using info from cursor
  prediction_step = pred_step(i)
  [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);

  % Calculating extreme learning machines values
  nn.w = calc_elm_weigths(train_set, target, regularization, nn)';

  % Apply network to all traces
  deconvolved_matrix(:, i) = target - neural_nete(train_set, nn);
  mse_prediction(i) = mean(deconvolved_matrix(:, i).^2);
  mse_reference_trace(i) = mean((deconvolved_matrix(:, i) - trace_norm_prim).^2);
  deconvolved_matrix(:, i) = deconvolved_matrix(:, i)*max_amp*std_dev + avg;

end

primaries = forward_radon_freq(traces_matrix_prim,dt,h_fo150,q,1,flow,fhigh);
prim_est = forward_radon_freq(deconvolved_matrix,dt,h_fo150,q,1,flow,fhigh);
multiples = forward_radon_freq(traces_matrix,dt,h_fo150,q,1,flow,fhigh);

primaries = traces_matrix_prim;
prim_est = deconvolved_matrix;
multiples = traces_matrix;

trace_nb_plot = size(traces_matrix,2);

% Plotting processed traces image
figure(1)
wiggle(primaries(1:trace_nb_plot, :), wiggle_th)
title('Traço somente primárias (após transformada Radon)')
grid
%saveas(gcf, sprintf('traco_primarias_%s.png', lower(data_case)));

figure(2)
title('Traço Original e filtrado (após a remoção das múltiplas')
subplot(1,2,1)
wiggle(multiples(1:trace_nb_plot, :), wiggle_th)
grid
%saveas(gcf, sprintf('traco_multiplas_%s.png', lower(data_case)));

subplot(1,2,2)
wiggle(prim_est(1:trace_nb_plot, :), wiggle_th)
grid
%saveas(gcf, sprintf('traco_filtrado_%s.png', lower(data_case)));

figure(4)
wiggle(prim_est(1:trace_nb_plot, :), wiggle_th)
title('Traço filtrado')
grid
%saveas(gcf, sprintf('traco1_tempo_%s.png', lower(data_case)));

figure(5)
wiggle(multiples(1:trace_nb_plot, :), wiggle_th)
title('Traço original')
grid

%save(lower(data_case), 'primaries', 'prim_est', 'multiples', 'nn', 'data_case')