
addpath('./DataSets')


ref_data_set_name = '../SyntheticTrace/trace_example_radon_domain';
load(ref_data_set_name)
data_set_name = 'trace_22_predict_sample_35';
load(data_set_name)


legends_name = zeros(1, length(filter_len));
figure(1)
hold on
for i = 1:length(filter_len)
    plot(mid_layer_sz, mse(1, :, i), '-.', 'LineWidth', 2)
end
legend(num2str(filter_len'))
ylabel('MSE')
xlabel('Neural network hidden layer size.')
grid

%%

reference_trace = trace_pre_processing(radon_mult_fo150, trace_nb, samples_start, attenuation_factor);

mid_layer_sz_plot = 51;
mid_layer_plot_idx = find(mid_layer_sz == mid_layer_sz_plot);


% plot(target)
% hold on
plot( -predicted_trace(:, mid_layer_plot_idx, 10))