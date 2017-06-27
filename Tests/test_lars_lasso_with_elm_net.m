
% Getting midlayer output matrix

rH] = get_elm_net_regression_matrix(train_set, target, nn);

% Removing bias from linear regression
Hr = H(:, 2:end);

[beta_hist, beta_sum] = lars_lasso(Hr, target', 1);

% Getting traces estimations
trace_estimation = H*beta_hist;

mse_error = mean((repmat(trace_norm, 1, size(trace_estimation, 2)) - trace_estimation).^2, 1);
mse_p_error = mean((repmat(trace_norm_prim, 1, size(trace_estimation, 2)) - (repmat(trace_norm, 1, size(trace_estimation, 2)) - trace_estimation)).^2, 1);



