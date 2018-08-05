function [posterior_prob, posterior] = gmm_posterior(train_set, mix_prob, mix_cov, mix_mean)
  number_of_components = length(mix_prob);

  posterior = zeros(size(train_set, 1), number_of_components);
  for i = 1:number_of_components
    posterior(:, i) = mix_prob(i)*mvgauss(train_set, mix_mean(:, :, i), mix_cov(:, :, i));
  end

  posterior_prob = posterior./repmat(sum(posterior, 2), 1, number_of_components);

end