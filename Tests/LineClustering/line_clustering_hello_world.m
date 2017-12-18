%% Line Clustering - Hello World

x = -2:0.1:4;
y1 = 1*x + 3;
y2 = 6*x - 2;

test_line1 = [x; y1];
test_line2 = [x; y2];

test_points_noise1 = test_line1 + 0.2*randn(size(test_line1));
test_points_noise2 = test_line2 + 0.2*randn(size(test_line2));
test_points = [test_points_noise1, test_points_noise2];

figure(1)
plot(test_line1(1, :), test_line1(2, :), test_line2(1, :), test_line2(2, :))
title('Testing reference lines')
xlim([-10 10])
ylim([-10 10])
grid

figure(2)
plot(test_points_noise1(1, :), test_points_noise1(2, :), test_points_noise2(1, :), test_points_noise2(2, :))
title('Noisy reference lines')
xlim([-10 10])
ylim([-10 10])
grid

%%
% Initial lines

init_line1.v = [1, 6];
init_line1.p = [0, 0];

init_line2.v = [1, 2];
init_line2.p = [0, -1];

range = -1:0.1:4;
init_line1_y = get_line_range(range, init_line1);
init_line2_y = get_line_range(range, init_line2);

figure(3)
plot(test_points_noise1(1, :), test_points_noise1(2, :), test_points_noise2(1, :), test_points_noise2(2, :))
hold on
plot(init_line1_y(:, 1), init_line1_y(:, 2), 'r', init_line2_y(:, 1), init_line2_y(:, 2), 'b')
xlim([-10 10])
ylim([-10 10])
grid

lines(1) = init_line1;
lines(2) = init_line2;

lines_number = 2;
coordenates_number = 2;
lines_points_number = 2;
points_number = size(test_points, 2);

%% Calculating points - line distance 

% Each run is a full iteration 
distance_matrix = zeros(lines_number, points_number);

for i = 1:points_number
  for j = 1:lines_number
    [normal_vectors(j, i), ~, distance_matrix(j, i)] = get_normal_vector(test_points(:, i)', lines(j));
  end
end

[~, line_owner] = min(distance_matrix);

% Plotting points and their normal vector

normal_vectors_line1 = normal_vectors(1, line_owner == 1);
normal_vectors_line2 = normal_vectors(2, line_owner == 2);

normal_vec = arrayfun(@(x)(x.p), normal_vectors_line1, 'UniformOutput', false);
normal_vectors_start1 = reshape([normal_vec{:}], coordenates_number, length(normal_vectors_line1))';
normal_vec = arrayfun(@(x)(x.p), normal_vectors_line2, 'UniformOutput', false);
normal_vectors_start2 = reshape([normal_vec{:}], coordenates_number, length(normal_vectors_line2))';
normal_vec = arrayfun(@(x)(x.v + x.p), normal_vectors_line1, 'UniformOutput', false);
normal_vectors_end1 = reshape([normal_vec{:}], coordenates_number, length(normal_vectors_line1))';
normal_vec = arrayfun(@(x)(x.v + x.p), normal_vectors_line2, 'UniformOutput', false);
normal_vectors_end2 = reshape([normal_vec{:}], coordenates_number, length(normal_vectors_line2))';

range = -1:0.1:4;
line1_pl = get_line_range(range, lines(1));
line2_pl = get_line_range(range, lines(2));

figure(4)
plot(test_points(1, line_owner == 1), test_points(2, line_owner == 1), '*r')
hold on
plot(test_points(1, line_owner == 2), test_points(2, line_owner == 2), '*b')
plot(line1_pl(:, 1), line1_pl(:, 2), 'r', line2_pl(:, 1), line2_pl(:, 2), 'b')
for i = 1:sum(line_owner == 1)
  plot([normal_vectors_start1(i, 1); normal_vectors_end1(i, 1)], [normal_vectors_start1(i, 2); normal_vectors_end1(i, 2)], 'r*--')
end
for i = 1:sum(line_owner == 2)
  plot([normal_vectors_start2(i, 1); normal_vectors_end2(i, 1)], [normal_vectors_start2(i, 2); normal_vectors_end2(i, 2)], 'b*--')
end
xlim([-10 10])
ylim([-10 10])
grid

% Calculation the normal average vector

for j = 1:lines_number
  normal_vec = arrayfun(@(x)(x.v), normal_vectors(j, line_owner == j), 'UniformOutput', false);
  normal_vec = reshape([normal_vec{:}], coordenates_number, length(normal_vectors(j, line_owner == j)))';
  avg_normal(j).v = mean(normal_vec, 1);
end

alpha = 0.5;
for j = 1:lines_number
  lines(j).p = lines(j).p + alpha*avg_normal(j).v;
end
