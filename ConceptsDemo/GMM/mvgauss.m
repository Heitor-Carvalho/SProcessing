function [prob] = mvgauss(x, mu, cov)

  % Data length 
 data_len = size(x, 1);

 % Removing mu
 x_avg = x - repmat(mu, data_len, 1);
 prob = (1/(((2*pi)^(length(mu)/2))*sqrt(abs(det(cov)))))*exp(-0.5*sum((x_avg*inv(cov + 0*eye(size(cov)))).*x_avg, 2));


end