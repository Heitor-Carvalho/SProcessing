function [mix_prob, mix_cov, mix_mean] = tstudentmm_em(train_set, mix_prob, mix_cov, mix_mean, v, reg, max_it)
  it = 0;
  mix_prob_prev = zeros(size(mix_prob));
  number_of_components = length(mix_prob);
  D = size(train_set, 2);
  for i = 1:number_of_components
    mix_cov(:, :, i)  = mix_cov(:, :, i) + reg*eye(size(mix_cov(:, :, i)));
  end
    
  while(it < max_it && max(mix_prob - mix_prob_prev) > 1e-4)
    mix_prob_prev = mix_prob;
    
    r = zeros(size(train_set, 1), number_of_components);
    u = zeros(size(train_set, 1), number_of_components);
    r_sum = zeros(size(train_set, 1), 1);

    % E - Step

    for i = 1:number_of_components
      % Conditional probability
      
      % Removing mean from data
      train_set_avg = train_set - repmat(mix_mean(:, :, i), size(train_set, 1), 1);
      
      % Sclate data to use mvtpdf
      scale = sqrt(diag(mix_cov(:,:,i)))';
      train_set_scale = train_set_avg./repmat(scale, size(train_set,1), 1);
      r(:, i) = mix_prob(i).*mvtpdf(train_set_scale, mix_cov(:, :, i), v);

      % Unscale data
      r(:, i) = r(:, i)/prod(scale);
      r_sum = r_sum + r(:, i);

      % Mahalanobis distance
      mahala_dist = sum((train_set_avg*inv(mix_cov(:, :, i))).*train_set_avg, 2);
      u(:, i) = (v + D)./(v + mahala_dist);
    end

    r = r./repmat(r_sum, 1, number_of_components);

    % M - Step
    mix_prob = mean(r, 1);

    for i = 1:number_of_components
      mix_mean(:, :, i) = sum(repmat(r(:, i), 1, size(train_set, 2)).*repmat(u(:, i), 1, size(train_set, 2)).*train_set)/sum(r(:, i).*u(:, i));
      mix_cov(:, :, i) = (repmat(r(:, i), 1, size(train_set, 2)).*repmat(u(:, i), 1, size(train_set, 2)).*train_set_avg)'*(train_set_avg)/sum(r(:, i)) ...
                        + reg*eye(size(mix_cov(:, :, i)));
    end

    it = it + 1;
  end
end
