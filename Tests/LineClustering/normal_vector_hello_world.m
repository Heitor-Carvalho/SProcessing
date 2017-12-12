%% Normal vector Hello!

% Let's define some line:
line1 = [0, 1; 
         0, 9];

line2 = [0, 1;
         0, 1];

% Test point
p = [0.34 -0.59]';

[normal_vector1, point_projection1, dist1] = get_normal_vector(p, line1);
[normal_vector2, point_projection2, dist2] = get_normal_vector(p, line2);

range = -1:0.1:1;
line_y_1 = get_line_range(range, line1);
line_y_2 = get_line_range(range, line2);
normal_vector_tip1 = p - normal_vector1;
normal_vector_tip2 = p - normal_vector2;

figure(1)
plot(range, line_y_1, 'b', range, line_y_2, 'r')
hold on
plot(p(1), p(2), 'o', point_projection1(1), point_projection1(2), '*', point_projection2(1), point_projection2(2), '*')
plot([p(1), normal_vector_tip1(1)], [p(2) normal_vector_tip1(2)],'--b')
plot([p(1), normal_vector_tip2(1)], [p(2) normal_vector_tip2(2)],'--r')
xlim([min(range) max(range)])
ylim([min(range) max(range)])
grid
