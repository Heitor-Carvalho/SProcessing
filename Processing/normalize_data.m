function [data] = normalize_data(data)

  data = data - mean(data);
  data = data/std(data);
  
end