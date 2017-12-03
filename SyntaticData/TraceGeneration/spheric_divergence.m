function [trace_div] = spheric_divergence(time, offsets, velocity, trace)
  div_esf = (1./(time*velocity)).';
  div_esf(1) = 1;
  div_esf = div_esf*ones(1, length(offsets));
  trace_div = trace.*div_esf;
end
