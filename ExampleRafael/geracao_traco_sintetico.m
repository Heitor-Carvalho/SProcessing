clear all;
close all;

addpath('../ThirdParty/SeismicLab/codes/synthetics/')

%%% Script para criar um cmp sintetico simulando multiplas de curto periodo
% Rafael Ferrari 06/08/2014

N_hip=25;
coef_ref=0.6;
t0_fundo_mar=0.1;
tau_1500=t0_fundo_mar*(1:1:N_hip).';
amp_1500=((-coef_ref).^(0:1:N_hip-1)).';
vel_1500=1500*ones(length(tau_1500),1);

% Frequencia central da wavelet
f=30;

% Vetor de offsets em metros
h=0:5:3500;

% Tempo maximo da aquisicao em segundos
tmax=3;

% Intervalo de amostragem em segundos
dt=0.004;

% SNR
SNR=1e8;

% reflexoes primarias
vel_p=1500;
tau_p=tau_1500(1);
amp_p=amp_1500(1);

% primarias
[dado_p,offsets,tempo] = hyperbolic_events(dt,f,tmax,h,tau_p,vel_p,amp_p,SNR,1);

% dado completo
[dado] = hyperbolic_events(dt,f,tmax,h,tau_1500,vel_1500,amp_1500,SNR,1);

figure(1); imagesc(offsets,tempo,dado);
figure(2); imagesc(offsets,tempo,dado_p);

% Calculo e aplicacao da divergencia esferica
div_esf = (1./(tempo*1500)).';
div_esf(1) = 1;

Div_esf = div_esf*ones(1,length(offsets));

dado_div = dado.*Div_esf;
dado_p_div = dado_p.*Div_esf;

figure(3) 
plot(tempo(1:400),dado(1:400,22)/max(dado(1:400,22)),'k');
title('Traço original')
grid

figure(4)
plot(tempo(1:400),dado_div(1:400,22)/max(dado_div(1:400,22)),'b');
title('Traço com compensação de divergência esférica')
grid

% Geração do traço com duas primárias para diferentes deslocamentos
primary_shift = [100, 75, 50, 20];

dado_div_two_primaries = zeros([size(dado_div) length(primary_shift)]);
for i = 1:length(primary_shift)
  for j = 1:size(dado_div, 2)
    dado_div_two_primaries(:, j, i) = dado_div(:, j) + circshift(dado_div(:, j), primary_shift(i));
  end
end

% Grafico com duas primarias - Traço 22 - Deslocamento de N amostras
for i = 1:length(primary_shift)
  figure(i+4)
  plot(dado_div_two_primaries(:, 22, i)/max(dado_div_two_primaries(:, 22, 2)), 'k')
  legend(sprintf('Deslocamento de %d amostras', primary_shift(i)))
  grid
end    

% Calcula dos traços no domínio transformado
flow = 3; 
fhigh = 80; 
mu = .010;
sol = 'ls';
radon_type = 1; 
q = linspace(0,7e-4,length(h));
h_fo150 = h(31:end);

radon_prim = zeros(size(dado_p_div));
radon_prim_offset150m = zeros(size(dado_p_div));
radon_mult = zeros(size(dado_div));
radon_mult_offset150m = zeros(size(dado_div));
radon_mult_prim = zeros(size(dado_div_two_primaries));
radon_mult_prim_offset150m = zeros(size(dado_div_two_primaries));

% Traço sem multiplas - Radon Domain
radon_prim = inverse_radon_freq(dado_p_div,dt,h,q,radon_type,flow,fhigh,mu,sol);
radon_prim_offset150m = inverse_radon_freq(dado_p_div(:, 31:end),dt,h_fo150,q,radon_type,flow,fhigh,mu,sol);

% Traço com multipla - Radon Domain
radon_mult = inverse_radon_freq(dado_div,dt,h,q,radon_type,flow,fhigh,mu,sol);
radon_mult_offset150m = inverse_radon_freq(dado_div(:, 31:end),dt,h_fo150,q,radon_type,flow,fhigh,mu,sol);

% Traço com  duas primarias e miltipla - Radon Domain
for i = 1:length(primary_shift)
  radon_mult_prim(:, :, i) = inverse_radon_freq(dado_div_two_primaries(:, :, i),dt,h,q,radon_type,flow,fhigh,mu,sol);
  radon_mult_prim_offset150m(:, :, i) = inverse_radon_freq(dado_div_two_primaries(:, 31:end, i),dt,h_fo150,q,radon_type,flow,fhigh,mu,sol);
end
    
    
save('tracos_tempo', 'dado_p_div', 'dado_div', 'dado_div_two_primaries');
save('tracos_radon', 'radon_prim', 'radon_prim_offset150m', ...
                     'radon_mult', 'radon_mult_offset150m', ...
                     'radon_mult_prim', 'radon_mult_prim_offset150m');
save('traco_parametros', 'dt', 'h', 'primary_shift');
    