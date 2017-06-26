% Loading traces matrix in Radon domain
data_set_name = './DataSets/tracos_radon_p2';
data_set_parm_name = './DataSets/traco_parametros';
load(data_set_name)
load(data_set_parm_name)

addpath('../ThirdParty/SeismicLab/codes/radon_transforms/')


%%

% Traces pre-processing
attenuation_factor = 1;
samples_start = 1;
traces_matrix = radon_mult_offset150m_p2;
traces_matrix_prim = radon_prim_offset150m_p2;

prediction_step = 17;
filter_len = 7;   
mid_layer_sz = 55;
regularization = 0;
initial_weigths_amp = 0.5;

deconvolved_matrix = zeros(size(traces_matrix));
trace_nb = size(traces_matrix, 2);

for i=1:trace_nb
  
  % Nomalizing data
  [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, i, samples_start, attenuation_factor);

  % Neural network setup
  clear nn
  in_sz = filter_len;
  out_sz = 1;
  nn.func = @tanh;
  nn.b = 0;

  nn.v = initial_weigths_amp*(rand(in_sz+1, mid_layer_sz));
  nn = neuro_net_init(nn);

  % Preparing data based in parameters
  [train_set, target] = trace_to_datatraining(trace_norm, filter_len, prediction_step);

  % Calculating extreme learning machines values
  nn.w = calc_elm_weigths(train_set, target, regularization, nn)';

  % Apply network to all traces
  deconvolved_matrix(:, i) = target - neural_nete(train_set, nn);
  deconvolved_matrix(:, i) = deconvolved_matrix(:, i)*max_amp*std_dev + avg;

end

primaries = forward_radon_freq(traces_matrix_prim,dt,h_fo150,q,1,flow,fhigh);
prim_est = forward_radon_freq(deconvolved_matrix,dt,h_fo150,q,1,flow,fhigh);
multiples = forward_radon_freq(traces_matrix,dt,h_fo150,q,1,flow,fhigh);

