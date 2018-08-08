clear all;
close all;

addpath('../../ThirdParty/SeismicLab/codes/synthetics/')
addpath('../../ThirdParty/SeismicLab/codes/radon_transforms/')
addpath('../../ThirdParty/SegyMAT/')

%% Zero offset Data
dt = 0.001;                         % Sampling interval in seconds
tmax = 3;                           % Max simulation time
time = 0:dt:tmax;                   % Time vector
reflextions_number = 20;            % Number of reflextions
max_idx = round(tmax/dt) + 1;       % Max time array index
tmax_plot = 1.5;                    % Max time in plot
wiggle_xlimit = 300;                 % Max trace in Wiggle plot x axis
wiggle_ylimit = 900;                 % Max trace in Wiggle plot y axis

% General simulation configurations
f = 30;                             % Wavelet central frequency
h = 0:5:3500;                       % Vetor de offsets em metros
SNR = 1e8;                          % SNR

%% Multiple response (water botton)
coef_ref_water = 0.6;
t0_water = 0.1;

% First layer (water botton) multiple response
multiple_amp = ((-coef_ref_water).^(0:1:reflextions_number-1)).';
multiple_delta_times = t0_water*(0:1:reflextions_number-1).';

% Removing reflections after tmax
multiple_amp(multiple_delta_times > tmax) = [];
multiple_delta_times(multiple_delta_times > tmax) = [];

multiple_idx = round(multiple_delta_times/dt)+1;
multiple_time_response = zeros(size(time));
multiple_time_response(multiple_idx) = multiple_amp;

%% First primary - Fist order multiples
t0_p1 = 0.1;
p1_amp = 1;
p1_delta_time = t0_p1;
p1_idx = round(p1_delta_time/dt)+1;

p1_time_response = zeros(size(time));
p1_time_response(p1_idx) = p1_amp;

p1_fst_mult_time_response = conv(p1_time_response, multiple_time_response);
p1_fst_mult_time_response = p1_fst_mult_time_response(1:max_idx);

p1_fst_mult_idx = find(p1_fst_mult_time_response ~= 0);
p1_fst_mult_amp = p1_fst_mult_time_response(p1_fst_mult_idx);
p1_fst_mult_delta_times = time(p1_fst_mult_idx);

% Removing primary
p1_fst_mult_time_response(p1_fst_mult_idx(1)) = 0;
p1_fst_mult_amp(1) = [];
p1_fst_mult_delta_times(1) = [];
p1_fst_mult_idx(1) = [];

%% Second primary - First order multiples
t0_p2 = 0.1;
p2_amp = 0.3;
p2_delta_time = t0_p2;
p2_idx = round(t0_p2/dt);

p2_time_response = zeros(size(time));
p2_time_response(p2_idx) = p2_amp;

p2_fst_mult_time_response = conv(p2_time_response, multiple_time_response);
p2_fst_mult_time_response = p2_fst_mult_time_response(1:max_idx);

p2_fst_mult_idx = find(p2_fst_mult_time_response ~= 0);
p2_fst_mult_amp = p2_fst_mult_time_response(p2_fst_mult_idx);
p2_fst_mult_delta_times = time(p2_fst_mult_idx);

% Removing primary
p2_fst_mult_time_response(p2_fst_mult_idx(1)) = 0;
p2_fst_mult_amp(1) = [];
p2_fst_mult_delta_times(1) = [];
p2_fst_mult_idx(1) = [];

%% Second order multiples - second primary
p2_sec_mult_time_response = conv(conv(p2_time_response, multiple_time_response), multiple_time_response);
p2_sec_mult_time_response = p2_sec_mult_time_response(1:max_idx);

p2_sec_mult_idx = find(p2_sec_mult_time_response ~= 0);
p2_sec_mult_amp = p2_sec_mult_time_response(p2_sec_mult_idx);
p2_sec_mult_delta_times = time(p2_sec_mult_idx);

% Removing primary
p2_sec_mult_time_response(p2_sec_mult_idx(1)) = 0;
p2_sec_mult_amp(1) = [];
p2_sec_mult_delta_times(1) = [];
p2_sec_mult_idx(1) = [];

%% All traces generations
hyperbolic_resumed = @(tau, velocity, amp) hyperbolic_events(dt, f, tmax, h, tau, velocity, amp, SNR, 1);
std_dev = 0;
hyperbolic_resumed_time = @(tau, velocity, amp) hyperbolic_events_time(dt, f, tmax, h, tau, velocity, amp, std_dev, 1);

% Water botton layer traces (first primary)
v1 = 1500*ones(length(h), 1);
tau1 = p1_fst_mult_delta_times;
amp1 = 1;

[trace_p1_fst_primaries, offsets, time] = hyperbolic_resumed(p1_delta_time, v1(1), amp1);
[trace_p1_fst_multiples, ~, ~] =  hyperbolic_resumed(tau1, v1, p1_fst_mult_amp);

[trace_p1_fst_primaries_time, ~, ~] = hyperbolic_resumed_time(p1_delta_time, v1(1), amp1);
[trace_p1_fst_multiples_time, ~, ~] =  hyperbolic_resumed_time(tau1, v1, p1_fst_mult_amp);

% Spheric divergence
trace_p1_fst_primaries_div = spheric_divergence(time, offsets, v1(1), trace_p1_fst_primaries);
trace_p1_fst_multiples_div = spheric_divergence(time, offsets, v1(1), trace_p1_fst_multiples);

trace_p1_fst_primaries_div_time = spheric_divergence(time, offsets, v1(1), trace_p1_fst_primaries_time);
trace_p1_fst_multiples_div_time = spheric_divergence(time, offsets, v1(1), trace_p1_fst_multiples_time);

% Second primary traces
v2 = 1650*ones(length(h),1);
tau2 = p2_fst_mult_delta_times;
amp2 = 0.3;

[trace_p2_fst_primaries, offsets, time] = hyperbolic_resumed(p2_delta_time, v2(1), amp2);
[trace_p2_fst_multiples, ~, ~] =  hyperbolic_resumed(tau2, v2, p2_fst_mult_amp);

[trace_p2_fst_primaries_time, ~, ~] = hyperbolic_resumed_time(p2_delta_time, v2(1), amp2);
[trace_p2_fst_multiples_time, ~, ~] =  hyperbolic_resumed_time(tau2, v2, p2_fst_mult_amp);

% Spheric divergence
trace_p2_fst_primaries_div = spheric_divergence(time, offsets, v2(1), trace_p2_fst_primaries);
trace_p2_fst_multiples_div = spheric_divergence(time, offsets, v2(1), trace_p2_fst_multiples);

trace_p2_fst_primaries_div_time = spheric_divergence(time, offsets, v2(1), trace_p2_fst_primaries_time);
trace_p2_fst_multiples_div_time = spheric_divergence(time, offsets, v2(1), trace_p2_fst_multiples_time);

% Second primary traces
tau2_sec = p2_sec_mult_delta_times;
amp2_sec = p2_sec_mult_amp;

[trace_p2_sec_multiples, offsets, time] = hyperbolic_resumed(tau2_sec, v2, p2_sec_mult_amp);

[trace_p2_sec_multiples_time, offsets, time] = hyperbolic_resumed_time(tau2_sec, v2, p2_sec_mult_amp);

% Spheric divergence
trace_p2_sec_multiples_div = spheric_divergence(time, offsets, v2(1), trace_p2_sec_multiples);

trace_p2_sec_multiples_div_time = spheric_divergence(time, offsets, v2(1), trace_p2_sec_multiples_time);

%% All traces - Cases

% Case 1 - Only P1

% Case 2 - P1 and fisrt order multiples
trace_p1_fst_prim_multiples = trace_p1_fst_primaries + trace_p1_fst_multiples;
trace_p1_fst_prim_multiples_div = trace_p1_fst_primaries_div + trace_p1_fst_multiples_div;

trace_p1_fst_prim_multiples_time = trace_p1_fst_primaries_time + trace_p1_fst_multiples_time;
trace_p1_fst_prim_multiples_div_time = trace_p1_fst_primaries_div_time + trace_p1_fst_multiples_div_time;


% Case 3 - Only P2

% Case 4 - P2 and fisrt order multiples
trace_p2_fst_prim_multiples = trace_p2_fst_primaries + trace_p2_fst_multiples;
trace_p2_fst_prim_multiples_div = trace_p2_fst_primaries_div + trace_p2_fst_multiples_div;

trace_p2_fst_prim_multiples_time = trace_p2_fst_primaries_time + trace_p2_fst_multiples_time;
trace_p2_fst_prim_multiples_div_time = trace_p2_fst_primaries_div_time + trace_p2_fst_multiples_div_time;

% Case 5 - P1 and P2 fisrt order multiples
trace_p1p2_fst_prim_multiples = trace_p1_fst_primaries + trace_p1_fst_multiples + trace_p2_fst_primaries + trace_p2_fst_multiples;
trace_p1p2_fst_prim_multiples_div = trace_p1_fst_primaries_div + trace_p1_fst_multiples_div + trace_p2_fst_primaries_div + trace_p2_fst_multiples_div;

trace_p1p2_fst_prim_multiples_time = trace_p1_fst_primaries_time + trace_p1_fst_multiples_time + trace_p2_fst_primaries_time + trace_p2_fst_multiples_time;
trace_p1p2_fst_prim_multiples_div_time = trace_p1_fst_primaries_div_time + trace_p1_fst_multiples_div_time + trace_p2_fst_primaries_div_time + trace_p2_fst_multiples_div_time;

% Case 6 - P2 second order multiples
trace_p2_sec_prim_multiples = trace_p2_fst_primaries + trace_p2_sec_multiples;
trace_p2_sec_prim_multiples_div = trace_p2_fst_primaries_div + trace_p2_sec_multiples_div;

trace_p2_sec_prim_multiples_time = trace_p2_fst_primaries_time + trace_p2_sec_multiples_time;
trace_p2_sec_prim_multiples_div_time = trace_p2_fst_primaries_div_time + trace_p2_sec_multiples_div_time;

% Case 7 - P1 and P2 second order multiples
trace_p1p2_sec_prim_multiples = trace_p1_fst_primaries + trace_p1_fst_multiples + trace_p2_fst_primaries + trace_p2_sec_multiples;
trace_p1p2_sec_prim_multiples_div = trace_p1_fst_primaries_div + trace_p1_fst_multiples_div + trace_p2_fst_primaries_div + trace_p2_sec_multiples_div;

trace_p1p2_sec_prim_multiples_time = trace_p1_fst_primaries_time + trace_p1_fst_multiples_time + trace_p2_fst_primaries_time + trace_p2_sec_multiples_time;
trace_p1p2_sec_prim_multiples_div_time = trace_p1_fst_primaries_div_time + trace_p1_fst_multiples_div_time + trace_p2_fst_primaries_div + trace_p2_sec_multiples_div_time;

% Case 8 - P1 and P2 only primaries
trace_p1p2_fst_primaries = trace_p1_fst_primaries + trace_p2_fst_primaries;
trace_p1p2_fst_primaries_div = trace_p1_fst_primaries_div + trace_p2_fst_primaries_div;

trace_p1p2_fst_primaries_time = trace_p1_fst_primaries_time + trace_p2_fst_primaries_time;
trace_p1p2_fst_primaries_div_time = trace_p1_fst_primaries_div_time + trace_p2_fst_primaries_div;

%% Traces in Radon domain
flow = 3;
fhigh = 80;
mu = .010;
sol = 'ls';
radon_type = 1;
q = linspace(0,7e-4,length(h));
h_ofsset150m = h(31:end);

inverse_radon_freq_res = @(trace, h) inverse_radon_freq(trace, dt, h, q, radon_type, flow, fhigh, mu, sol);

% Case 1 - Only P1
radon_p1_fst_prim_div = inverse_radon_freq_res(trace_p1_fst_primaries_div, h);
radon_p1_fst_prim_div_offset = inverse_radon_freq_res(trace_p1_fst_primaries_div(:, 31:end), h_ofsset150m);

% Case 2 - P1 and fisrt order multiples
radon_p1_fst_mul_div = inverse_radon_freq_res(trace_p1_fst_prim_multiples_div, h);
radon_p1_fst_mul_div_offset = inverse_radon_freq_res(trace_p1_fst_prim_multiples_div(:, 31:end), h_ofsset150m);

% Case 3 - Only P2
radon_p2_fst_prim_div = inverse_radon_freq_res(trace_p2_fst_primaries_div, h);
radon_p2_fst_prim_div_offset = inverse_radon_freq_res(trace_p2_fst_primaries_div(:, 31:end), h_ofsset150m);

% Case 4 - P2 and fisrt order multiples
radon_p2_fst_mul_div = inverse_radon_freq_res(trace_p2_fst_prim_multiples_div, h);
radon_p2_fst_mul_div_offset = inverse_radon_freq_res(trace_p2_fst_prim_multiples_div(:, 31:end), h_ofsset150m);

% Case 5 - P1 and P2 fisrt order multiples
radon_p1p2_fst_mul_div = inverse_radon_freq_res(trace_p1p2_fst_prim_multiples_div, h);
radon_p1p2_fst_mul_div_offset = inverse_radon_freq_res(trace_p1p2_fst_prim_multiples_div(:, 31:end), h_ofsset150m);

% Case 6 - P2 second order multiples
radon_p2_sec_mul_div = inverse_radon_freq_res(trace_p2_sec_prim_multiples_div, h);
radon_p2_sec_mul_div_offset = inverse_radon_freq_res(trace_p2_sec_prim_multiples_div(:, 31:end), h_ofsset150m);

% Case 7 - P1 and P2 second order multiples
radon_p1p2_sec_mul_div = inverse_radon_freq_res(trace_p1p2_sec_prim_multiples_div, h);
radon_p1p2_sec_mul_div_offset = inverse_radon_freq_res(trace_p1p2_sec_prim_multiples_div(:, 31:end), h_ofsset150m);

% Case 8 - P1 and P2 primries
radon_p1p2_primaries_div = inverse_radon_freq_res(trace_p1p2_fst_primaries_div, h);
radon_p1p2_primaries_div_offset = inverse_radon_freq_res(trace_p1p2_fst_primaries_div(:, 31:end), h_ofsset150m);

%% Saving data
save('tracos_in_time', 'trace_p1_fst_primaries_div'       , 'trace_p1_fst_primaries'  , ...
                       'trace_p1_fst_prim_multiples_div'  , 'trace_p1_fst_prim_multiples'  , ...
                       'trace_p2_fst_primaries_div'       , 'trace_p2_fst_primaries'       , ...
                       'trace_p2_fst_prim_multiples_div'  , 'trace_p2_fst_prim_multiples'  , ...
                       'trace_p1p2_fst_prim_multiples_div', 'trace_p1p2_fst_prim_multiples', ...
                       'trace_p2_sec_prim_multiples_div'  , 'trace_p2_sec_prim_multiples'  , ...
                       'trace_p1p2_sec_prim_multiples_div', 'trace_p1p2_sec_prim_multiples', ...
                       'trace_p1p2_fst_primaries_div'     , 'trace_p1p2_fst_primaries');

save('tracos_in_time_ideal', 'trace_p1_fst_primaries_div_time'       , 'trace_p1_fst_prim_multiples_time'  , ...
                             'trace_p1_fst_prim_multiples_div_time'  , 'trace_p1_fst_prim_multiples_time'  , ...
                             'trace_p2_fst_primaries_div_time'       , 'trace_p2_fst_primaries_time'       , ...
                             'trace_p2_fst_prim_multiples_div_time'  , 'trace_p2_fst_prim_multiples_time'  , ...
                             'trace_p1p2_fst_prim_multiples_div_time', 'trace_p1p2_fst_prim_multiples_time', ...
                             'trace_p2_sec_prim_multiples_div_time'  , 'trace_p2_sec_prim_multiples_time'  , ...
                             'trace_p1p2_sec_prim_multiples_div_time', 'trace_p1p2_sec_prim_multiples_time', ...
                             'trace_p1p2_fst_primaries_div_time'     , 'trace_p1p2_fst_primaries_time');

save('tracos_in_radon', 'radon_p1_fst_prim_div'   , 'radon_p1_fst_prim_div_offset'   , ...
                        'radon_p1_fst_mul_div'    , 'radon_p1_fst_mul_div_offset'    , ...
                        'radon_p2_fst_prim_div'   , 'radon_p2_fst_prim_div_offset'   , ...
                        'radon_p2_fst_mul_div'    , 'radon_p2_fst_mul_div_offset'    , ...
                        'radon_p1p2_fst_mul_div'  , 'radon_p1p2_fst_mul_div_offset'  , ...
                        'radon_p2_sec_mul_div'    , 'radon_p2_sec_mul_div_offset'    , ...
                        'radon_p1p2_sec_mul_div'  , 'radon_p1p2_sec_mul_div_offset'  , ...
                        'radon_p1p2_primaries_div', 'radon_p1p2_primaries_div_offset');

save('parameter', 'dt', 'tmax', 'reflextions_number', 'f', 'h', 'SNR', ...
                  'coef_ref_water', 't0_water', 't0_p1', 'p1_amp'    , ...
                  't0_p2', 'p2_amp', 'flow', 'fhigh', 'mu'           , ...
                  'radon_type', 'q', 'h_ofsset150m', 'v1', 'v2')
%%
close all
