%% Normal vector Hello!

clear all
clc 

% Let's define some line:
line1.v = [1, 2];
line1.p = [1, 1];

% Test point
p = [3, 4];

% Plotting lines and point
range = -1:0.1:4;
line_arr_1 = get_line_range(range, line1);

[normal_vector, projection, dist] = get_normal_vector(p, line1)
projection_end = projection.v + projection.p;
normal_vector_end = normal_vector.v + normal_vector.p;

figure(1)
plot(line_arr_1(:, 1), line_arr_1(:, 2), 'b')
hold on
plot(p(1), p(2), 'k*')
plot([projection.p(1), projection_end(1)], [projection.p(2), projection_end(2)], '--g^') 
plot([normal_vector.p(1), normal_vector_end(1)], [normal_vector.p(2), normal_vector_end(2)],'--m^') 
xlim([0 6])
ylim([0 6])
grid

% In this code, a line is defined as a vector on top of a point
line1.p = [0, 2];
line_arr_1_shfited = get_line_range(range, line1);
figure(2)
plot(line_arr_1(:, 1), line_arr_1(:, 2), 'b')
hold on
plot(line_arr_1_shfited(:, 1), line_arr_1_shfited(:, 2), 'b')
xlim([0 6])
ylim([0 6])
grid

