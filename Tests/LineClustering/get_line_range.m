function [line_array, range] = get_line_range(range, line)
  line_array = repmat(line.v, length(range), 1).*repmat(range, length(line.v), 1)' + repmat(line.p, length(range), 1);
end