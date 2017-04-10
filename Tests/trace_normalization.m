function [trace_norm] = trace_normalization(trace)
% [trace_norm] = trace_normalization(trace) - Removes the signal
% average and divides by the variance.
% Inputs:
%  trace - Vector with trace to be normalized
% Outputs:
%  trace_norm - normalized trace

trace_norm = (trace-mean(trace))/std(trace);

end