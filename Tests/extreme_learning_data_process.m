clear all
addpath('./DataSets')

ref_data_set_name = './DataSets/tracos_radon';
load(ref_data_set_name)

trace_predicted = 22;
sample_predicted = 33;
data_set_name = sprintf('trace_%d_predict_sample_%d', trace_predicted, sample_predicted);
load(data_set_name)

%% MSE versus filter length and neural network middle layer analisys 

legends_name = zeros(1, length(filter_len));
figure
hold on
for i = 1:length(filter_len)
    plot(mid_layer_sz, mse(1, :, i), '-.', 'LineWidth', 2)
end
legend(num2str(filter_len'))
ylabel('MSE')
xlabel('Neural network hidden layer size.')
grid

% saveas(gcf, sprintf('MSE_NetworkSize_X_FilterLenSample%d.png', sample_predicted))

%% Reference trace versus predicted trace analisys 

reference_trace = trace_pre_processing(radon_mult_offset150m, trace_nb, samples_start, attenuation_factor);

mid_layer_sz_plot = 69;
mid_layer_plot_idx = find(mid_layer_sz == mid_layer_sz_plot);

predictedXreferece_plot_name     = 'PredictedTrace_X_RefereceTrace_filter_length%d_mid_layer_size%d.png';
predictedMinusreferece_plot_name = 'PredictedTrace_Minus_RefereceTrace_filter_length%d_mid_layer_size%d.png';
max_sample = 300;

for i = 1:length(filter_len)
    
    [train_set, target] = trace_to_datatraining(reference_trace, filter_len(i), sample_to_predict-filter_len(i));

    figure
    plot(target, '--')
    hold on
    plot(predicted_trace(:, mid_layer_plot_idx, i), '.')
    title(sprintf('Filter length %d - Neural network size %d', filter_len(i), mid_layer_sz_plot));
    legend('Prediction reference', 'Predicted Trace')
    xlim([0 max_sample]);
    grid

%     saveas(gcf, sprintf(predictedXreferece_plot_name, filter_len(i), mid_layer_sz_plot));
    
    figure
    trace = [reference_trace(1:sample_predicted); target'-predicted_trace(:, mid_layer_plot_idx, i)]; 
    trace = trace/max(trace);
    plot(radon_prim_offset150m(19:end, 22)./max(radon_prim_offset150m(19:end, 22)),'r--')
    hold on
    plot(trace)
    ylim([min(trace) max(trace)])
    xlim([0 110]);
    legend(sprintf('Filter length %d - Neural network size %d', filter_len(i), mid_layer_sz_plot));
    grid

%     saveas(gcf, sprintf(predictedMinusreferece_plot_name, filter_len(i), mid_layer_sz_plot));
end
