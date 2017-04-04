function [mse_error, out, target] = test_midsize_neural_net(trace, min_sz, max_sz, filter_len, predictor_len, regularization) 

  % Preparing training set
  trace = normalize_data(trace);
  trace = trace/max(trace);
  target = trace(filter_len + predictor_len+1:end)'; 
  train_set = zeros(filter_len, length(trace)-filter_len-predictor_len);
  for i = 1:length(trace)-filter_len-predictor_len
    train_set(1:filter_len, i) = trace(i:i+filter_len-1);
  end

  % MSE for different neural networks middle layer size
  mid_layer_sz = min_sz:max_sz;
  mse_error = zeros(length(mid_layer_sz), 1);
  nn.b = 1;
  nn.func = @(x) exp(x)./(1 + exp(x));
  nn.diff = @(x) exp(x)./(1 + exp(x)).^2;
  for i = 1:length(mid_layer_sz)
    nn.v = rand(filter_len+1, mid_layer_sz(i));
    nn.w = rand(1, mid_layer_sz(i)+1);
    nn = neuro_net_init(nn);
    nn.w = calc_elm_weigths(train_set, target, regularization, nn)';
    out = neural_nete(train_set, nn);
    mse_error(i) = mean((target - out).^2);
  end

end
