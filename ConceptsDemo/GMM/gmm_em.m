function [mix_prob, mix_cov, mix_mean] = gmm_em(train_set, mix_prob, mix_cov, mix_mean, reg, max_it)
  it = 0;
  mix_prob_prev = zeros(size(mix_prob));
  number_of_components = length(mix_prob);
    
  for i = 1:number_of_components
    mix_cov(:, :, i)  = mix_cov(:, :, i) + reg*eye(size(mix_cov(:, :, i)));
  end
  
  while(it < max_it && max(mix_prob - mix_prob_prev) > 1e-4)
    mix_prob_prev = mix_prob;

    r = zeros(size(train_set, 1), number_of_components);
    r_sum = zeros(size(train_set, 1), 1);

    % E - Step
    for i = 1:number_of_components
      r(:, i) = mix_prob(i).*mvgauss(train_set, mix_mean(:,:, i), mix_cov(:, :, i));
      r_sum = r_sum + r(:, i);
    end

    r = r./repmat(r_sum, 1, number_of_components);

    % M - Step
    mix_prob = mean(r, 1);
    for i = 1:number_of_components
      mix_mean(:, :, i) = sum(repmat(r(:, i), 1, size(train_set, 2)).*train_set)/sum(r(:, i));
      mix_cov(:, :, i) = ((repmat(r(:, i), 1, size(train_set, 2)).*train_set)'*train_set)/sum(r(:, i)) - mix_mean(:, :, i)'*mix_mean(:, :, i) + ...
                         reg*eye(size(mix_cov(:, :, i)));
    end

    it = it + 1;
  end
end
