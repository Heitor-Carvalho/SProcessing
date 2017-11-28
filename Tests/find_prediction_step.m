% Fitting a polynomial based in cursor_info points to predict
x = zeros(length(cursor_info), 1);
y = zeros(length(cursor_info), 1);
for i = 1:length(x)
  x(i) = cursor_info(i).Position(1);
  y(i) = cursor_info(i).Position(2);
end
 
A = [ones(size(x)) x x.^2];
coef = inv(A'*A)*A'*y;

x_test = 1:size(trace_matrix,2);
x_test = x_test';
A_test = [ones(size(x_test)) x_test x_test.^2];

multiple_event_idx = A_test*coef;

plot(x_test, multiple_event_idx , x, y, 'o')
title('Multiple Events - Data and fittied curve')
grid