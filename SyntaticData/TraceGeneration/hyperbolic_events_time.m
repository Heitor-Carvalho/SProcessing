function [trace_matrix, h, time] = hyperbolic_events_time(dt, f0, tmax, h, tau, v, amp, std_dev, L)

  max_idx = round(tmax/dt)+1;
  time = 0:dt:tmax;
  wavelet = ricker(f0,dt);
  wavelet_sz = length(wavelet);
  wavelet_delay_idx = round(wavelet_sz/2);

  offset_delay = zeros(length(tau), length(h));
  offset_delay_idx = zeros(length(tau), length(h));

  trace_matrix = zeros(max_idx, length(h));

  for k=1:length(h)
    offset_delay(:, k) = sqrt(tau.^2 + (h(k)./v(k)).^2);
    offset_delay_idx(:, k) = round(offset_delay(:, k)/dt);
    out_range_idx = offset_delay_idx(:,k) > max_idx;
    trace_matrix(offset_delay_idx(~out_range_idx,k)-wavelet_delay_idx, k) = amp(~out_range_idx);
    trace = conv(trace_matrix(:, k), wavelet);
    trace(max_idx+1:end) = [];
    trace_matrix(:, k) = trace + sqrt(std_dev)*randn(size(trace));;
  end

end
