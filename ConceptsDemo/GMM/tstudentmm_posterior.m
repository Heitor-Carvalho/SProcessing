function [posterior_prob, posterior] = tstudentmm_posterior(train_set, mix_prob, mix_cov, mix_mean, v)
  number_of_components = length(mix_prob);

  posterior = zeros(size(train_set, 1), number_of_components);
  for i = 1:number_of_components
    train_set_avg = train_set - repmat(mix_mean(:, :, i), size(train_set,1), 1);
    scale = sqrt(diag(mix_cov(:,:,i)))';
    train_set_scale = train_set_avg./repmat(scale, size(train_set,1), 1);
    posterior(:, i) = mix_prob(i)*mvtpdf(train_set_scale, mix_cov(:, :, i), v);
    posterior(:, i) = posterior(:, i)/prod(scale);
  end

  posterior_prob = posterior./repmat(sum(posterior, 2), 1, number_of_components);

end