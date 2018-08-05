%% Testing Multivariate gaussian function
clear all
close all

[x_grid, y_grid] = meshgrid(-5:0.1:5);

g_grid = [x_grid(:), y_grid(:)];

mu = [0, 0];
cov = diag([1.5 0.5]);

% Initialization
x = g_grid;

%% Function start

prob = mvgauss(x, mu, cov);

%% Plotting results

prob_grid = reshape(prob, size(x_grid));

figure(1)
mesh(x_grid, y_grid, prob_grid)
title('Multivariate gaussian test')
grid on

% Checking for integral equal 1
trapz(-5:0.1:5, trapz(-5:0.1:5, prob_grid, 2), 1)

%%
close all