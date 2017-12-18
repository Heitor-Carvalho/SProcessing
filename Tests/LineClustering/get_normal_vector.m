function [normal_vector, projection, dist] = get_normal_vector(point, line_point)

  % Moving point to origin
  point = point - line_point.p;
  
  % Projection point into line at origin
  projection.v = line_point.v*(line_point.v*point')/(line_point.v*line_point.v');
  projection.p = line_point.p;
  
  % Getting the normal vector at origin
  normal_vector.v = point - projection.v;
  normal_vector.p = projection.v + projection.p;
  
  % Distance = length(normal_vector)
  dist = sqrt(normal_vector.v*normal_vector.v');

end
