function [trace_norm, std_dev, avg, max_amp] = trace_pre_processing(traces_matrix, trace_nb, samples_start, attenuation_factor)
  
  trace = traces_matrix(:, trace_nb);
  [trace_norm, avg, std_dev] = trace_normalization(trace(samples_start:end));
  max_amp = max(trace_norm);
  trace_norm = attenuation_factor*trace_norm/max_amp;
  
end