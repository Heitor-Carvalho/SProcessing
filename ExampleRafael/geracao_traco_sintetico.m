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
plot(dado(:, 22))
title('Tra�os originais')

figure(4)
plot(dado_div(:, 22))
title('Tra�os com compensa��o de diverg�ncia esf�rica')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Geracao da segunda primaria e suas multiplas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t02 = 0.35; % tempo de transito da segunda primaria no offset 0m
v2 = 1650; % velocidade de empilhamento da segunda primaria
coef_ref2=0.3;
tau_v2 = tau_1500 + t02 - t0_fundo_mar;
amp_v2 = coef_ref2*amp_1500;
vel_v2= v2*ones(length(tau_v2),1);

% reflexoes primarias
vel_p=v2;
tau_p=tau_v2(1);
amp_p=amp_v2(1);

% primarias
[dado_p_v2,offsets,tempo] = hyperbolic_events(dt,f,tmax,h,tau_p,vel_p,amp_p,SNR,1);

% dado completo
[dado_v2] = hyperbolic_events(dt,f,tmax,h,tau_v2,vel_v2,amp_v2,SNR,1);

figure(5); imagesc(offsets,tempo,dado_v2);
figure(6); imagesc(offsets,tempo,dado_p_v2);

% Calculo e aplicacao da divergencia esferica
div_esf = (1./(tempo*v2)).';
div_esf(1) = 1;

Div_esf = div_esf*ones(1,length(offsets));

dado_div_v2 = dado_v2.*Div_esf;
dado_p_div_v2 = dado_p_v2.*Div_esf;

figure(7)
plot(dado_v2(:, 22))
title('Tra�os originais')

figure(8)
wigb(dado_div_v2(:, 22))
title('Tra�os com compensa��o de diverg�ncia esf�rica')

% Dado sem correcao de divergencia esferica
dado_completo = dado + dado_v2;

% Dado com divergencia esferica
dado_completo_div = dado_div + dado_div_v2;
dado_completo_primarias_div = dado_p_div + dado_p_div_v2;

figure(9)
plot(dado_completo(:, 22))
title('Tra�os originais')

figure(10)
plot(dado_completo_div(:, 22))
title('Tra�os com diverg�ncia esf�rica')

% Calculo dos tra�os no dom�nio transformado
flow = 3; 
fhigh = 80; 
mu = .010;
sol = 'ls';
radon_type = 1; 
q = linspace(0,7e-4,length(h));
h_fo150 = h(31:end);

% Tra�o sem multiplas - Radon Domain
radon_prim = inverse_radon_freq(dado_completo_primarias_div,dt,h,q,radon_type,flow,fhigh,mu,sol);
radon_prim_offset150m = inverse_radon_freq(dado_completo_primarias_div(:, 31:end),dt,h_fo150,q,radon_type,flow,fhigh,mu,sol);

% Tra�o com multiplas - Radon Domain
radon_mult = inverse_radon_freq(dado_completo_div,dt,h,q,radon_type,flow,fhigh,mu,sol);
radon_mult_offset150m = inverse_radon_freq(dado_completo_div(:, 31:end),dt,h_fo150,q,radon_type,flow,fhigh,mu,sol);

  
save('tracos_tempo', 'dado_completo_primarias_div', 'dado_completo_div');
save('tracos_radon', 'radon_prim', 'radon_prim_offset150m', ...
                     'radon_mult', 'radon_mult_offset150m');
save('traco_parametros', 'dt', 'h', 't0_fundo_mar', 't02', 'v2');
