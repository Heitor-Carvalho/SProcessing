function [trace_norm] = trace_pre_processing(traces_matrix, trace_nb, samples_start, attenuation_factor)
  
  trace = traces_matrix(:, trace_nb);
  trace_norm = trace_normalization(trace(samples_start:end));
  trace_norm = attenuation_factor*trace_norm/max(trace_norm);

end