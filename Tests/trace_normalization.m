function [trace_norm, avg, std_dev] = trace_normalization(trace)
% [trace_norm] = trace_normalization(trace) - Removes the signal
% average and divides by the variance.
% Inputs:
%  trace - Vector with trace to be normalized
% Outputs:
%  trace_norm - normalized trace

avg = mean(trace);
std_dev = std(trace);
trace_norm = (trace-avg)/std_dev;

end