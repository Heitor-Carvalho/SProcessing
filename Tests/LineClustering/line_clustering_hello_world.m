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
xlim([-1 4])
grid

figure(2)
plot(test_points_noise1(1, :), test_points_noise1(2, :), test_points_noise2(1, :), test_points_noise2(2, :))
title('Noisy reference lines')
xlim([-1 4])
grid


% Initial lines
init_line1 = [0, 1;
              0, 9];

init_line2 = [0, 1;
              0, 1];

range = -1:0.1:4;
init_line1_y = get_line_range(range, init_line1);
init_line2_y = get_line_range(range, init_line2);

figure(3)
plot(test_points_noise1(1, :), test_points_noise1(2, :), test_points_noise2(1, :), test_points_noise2(2, :))
hold on
plot(range, init_line1_y, range, init_line2_y )
xlim([-1 4])
grid

%%

% Calculate distance from all lines
lines_number = 2;
coordenates_number = 2;
lines_points_number = 2;
points_number = size(test_points, 2);

lines = zeros(coordenates_number, lines_points_number, lines_number);

lines(:, :, 1) = init_line1;
lines(:, :, 2) = init_line2;

distance_matrix = zeros(lines_number, points_number);
normal_vectors = zeros(coordenates_number, lines_number, points_number);

for i = 1:points_number
  for j = 1:lines_number
    [normal_vectors(:, j, i), ~, distance_matrix(j, i)] = get_normal_vector(test_points(:, i), lines(:,:,j));
  end
end

[~, line_owner] = min(distance_matrix);

normal_vectors_line1 = reshape(normal_vectors(:, 1, line_owner == 1), coordenates_number, sum(line_owner == 1));
normal_vectors_line2 = reshape(normal_vectors(:, 2, line_owner == 2), coordenates_number, sum(line_owner == 2));
normal_vectors_tip1 = test_points(:, line_owner == 1) - normal_vectors_line1;
normal_vectors_tip2 = test_points(:, line_owner == 2) - normal_vectors_line2;

figure(4)
plot(test_points(1, line_owner == 1), test_points(2, line_owner == 1), '*g')
hold on
plot(test_points(1, line_owner == 2), test_points(2, line_owner == 2), '*r')
plot(range, init_line1_y, 'g', range, init_line2_y, 'r')
idx = 1:length(line_owner);
idx_line1 = idx(line_owner == 1);
idx_line2 = idx(line_owner == 2);
for i = 1:sum(line_owner == 1)
plot([test_points(1, idx_line1(i)),  normal_vectors_tip1(1, i)], [test_points(2, idx_line1(i)),  normal_vectors_tip1(2, i)], '*--g')
end
for i = 1:sum(line_owner == 2)
  plot([test_points(1, idx_line2(i)),  normal_vectors_tip2(1, i)], [test_points(2, idx_line2(i)),  normal_vectors_tip2(2, i)], '*--r')
end
xlim([-10 10])
ylim([-10 10])
grid
