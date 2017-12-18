function [y_est, w, P] = rls(h, y, init_cov_matrix, w0, lambda)
  
  h = [zeros(size(h, 1), 1) h];
  y = [zeros(size(y, 1), 1), y];
  
  w = zeros(size(h));
  g = zeros(size(h));
  e = zeros(size(y));
  y_est = zeros(size(y));
  P = zeros(size(h, 1), size(h, 1), size(h, 2));
  P(:,:, 1) = init_cov_matrix;
  w(:, 1) = w0;

  for i = 1:length(h)-1
    y_est(i+1) = h(:, i+1)'*w(:, i);
    e(i+1) = y(i+1) - h(:, i+1)'*w(:, i);
    g(:, i+1) = P(:,:, i)*h(:, i+1).*(lambda + h(:, i+1)'*P(:,:, i)*h(:, i+1))^-1;
    P(:, :, i+1) = P(:,:, i)/lambda - g(:, i+1)*h(:, i+1)'*P(:,:, i)/lambda;
    w(:, i+1) = w(:, i) + g(:, i+1)*e(i+1);
  end
  
  y_est(1) = [];
  
end
