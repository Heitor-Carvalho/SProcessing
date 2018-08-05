function [pdf] = tstudentmm_pdf(train_set, mix_prob, mix_cov, mix_mean, v)
  number_of_components = length(mix_prob);

  pdf = zeros(size(train_set, 1), 1);
  for i = 1:number_of_components
    train_set_avg = train_set - repmat(mix_mean(:, :, i), size(train_set,1), 1);
    scale = sqrt(diag(mix_cov(:,:,i)))';
    train_set_scale = train_set_avg./repmat(scale, size(train_set,1), 1);
    pdf = pdf + mix_prob(i)*mvtpdf(train_set_scale, mix_cov(:, :, i), v);
  end

end