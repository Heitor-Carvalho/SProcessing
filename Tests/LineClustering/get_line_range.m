function [init_line1_y, range] = get_line_range(range, line_points)
   m = diff(line_points(2, :))/diff(line_points(1,:));
   init_line1_y = m*range + line_points(2, 1);
end