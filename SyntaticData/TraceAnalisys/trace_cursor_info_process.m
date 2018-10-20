%% Loading traces
clear all

path = '../../SyntaticData/SimulatedDataGeneration/SynData_035/';
data_set_name = 'SynData_035_offset';
load([path, 'tracos_in_radon']);
load([path, 'parameter']);

%% Plotting traces

% Ploting filtered trace and reference trace
traces_matrix = radon_p1p2_sec_mul_div_offset;
traces_matrix_prim = radon_p1p2_primaries_div_offset;


figure(1)
imagesc(traces_matrix, [-1 1]*0.5e-6)
ylim([0 500])
grid

figure(2)
imagesc(traces_matrix_prim, [-1 1]*0.5e-6)
ylim([0 500])
grid

%% Loading cursor data:

cursor_data = cursor_info;

data_points = zeros(length(cursor_data), 2);

for i = 1:length(cursor_data)
  data_points(i, :) = cursor_data(i).Position;
end

%% Plotting data points

line_regression_coef = pinv([data_points(:, 1) ones(size(data_points(:, 2)))])*data_points(:, 2);
data_points_line = [data_points(:, 1) ones(size(data_points(:, 1)))]*line_regression_coef;
xrange = 0:max(data_points(:, 1));

figure(3)
plot(data_points(:, 1), data_points(:, 2), 'o')
hold on
plot(xrange, [xrange' ones(size(xrange'))]*line_regression_coef)
plot(data_points(:, 1), data_points_line, 'or')
grid


upper_points = data_points(data_points(:, 2) > data_points_line, :);
lower_points = data_points(data_points(:, 2) < data_points_line, :);


%% Find a polinomial regression

upper_points_poly = [ones(size(upper_points(:, 1))) upper_points(:, 1) upper_points(:, 1).^2 upper_points(:, 1).^3 upper_points(:, 1).^4];
upper_poly_regression_coef = pinv(upper_points_poly)*upper_points(:, 2);
upper_points_aprox = upper_points_poly*upper_poly_regression_coef;

lower_points_poly = [ones(size(lower_points(:, 1))) lower_points(:, 1) lower_points(:, 1).^2 lower_points(:, 1).^3 lower_points(:, 1).^4];
lower_poly_regression_coef = pinv(lower_points_poly)*lower_points(:, 2);
lower_points_aprox = lower_points_poly*lower_poly_regression_coef;

trace_max = size(traces_matrix, 2);
trace_range = (1:trace_max)';
trace_poly = [ones(size(trace_range)), trace_range trace_range.^2 trace_range.^3 trace_range.^4];

upper_trace_aprox = round(trace_poly*upper_poly_regression_coef);
upper_trace_aprox = max(upper_trace_aprox , 1);

lower_trace_aprox = round(trace_poly*lower_poly_regression_coef);
lower_trace_aprox = max(lower_trace_aprox , 1);

figure(4)
plot(upper_points(:, 1), upper_points_aprox, 'o')
hold on
plot(trace_range, upper_trace_aprox, '--.')
plot(lower_points(:, 1), lower_points_aprox, 'o')
plot(trace_range, lower_trace_aprox, '--.')
grid

%% Calculating prediction steps
prediction_step = upper_trace_aprox - lower_trace_aprox;
prediction_step = max(prediction_step, 1);

figure(5)
plot(trace_range, prediction_step)
grid


save([data_set_name, 'prediction_step'], 'prediction_step')




