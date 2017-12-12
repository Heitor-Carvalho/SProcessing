function [normal_vector, point_projection, dist] = get_normal_vector(point, line_point)

  % Remove Y-axis offset
  line_point(2, 2) = line_point(2, 2) - line_point(2, 1);
  point(2) = point(2) - line_point(2, 1);

  % Projection point into line
  point_projection = line_point(:, 2)*(line_point(:, 2)'*point)/(line_point(:, 2)'*line_point(:, 2));

  % Getting the normal vector
  normal_vector = point - point_projection;

  point_projection(2) = point_projection(2) + line_point(2, 1);
  
  % Distance = length(normal_vector)
  dist = sqrt(normal_vector'*normal_vector);

end
