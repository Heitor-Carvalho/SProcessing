%% Loading data

addpath('../Tests');
addpath('../ThirdParty/SeismicLab/codes/radon_transforms/')
addpath('../ThirdParty/SegyMAT/')

%%
addpath('../../../IA353/NeuralNetwork/')
addpath('../../../IA353/ExtremeLearningMachine/')
addpath('../../../IA353/EchoStateNetworks/')
addpath('../../../IA353/Regularization/')

%% Vertical line


line_angle = zeros(10, 1); 
line_amp = linspace(0, 1, 10);

angles = (0:30:180)*pi/180;
lines_number = length(angles);

lines_angle = repmat(line_angle, 1, lines_number) + repmat(angles, length(line_angle), 1);
lines_amp = repmat(line_amp, 1, lines_number);

lines_angle = reshape(lines_angle, prod(size(lines_angle)), 1);
lines_amp = reshape(lines_amp, prod(size(lines_amp)), 1);

[x, y] = pol2cart(lines_angle, lines_amp)
plot(x, -y)





% amps = repmat(amp_base, length(rotation), 1);
% angles = repmat(angle_base, length(rotation), 1) + repmat(rotation', 1, length(angle_base));

% polar(lines_angle, lines_amp)
% xlim([0 100])

% s = ones(size(i));

% matrix = sparse(i, j, s)

