clear all;
close all;

addpath('../ThirdParty/SeismicLab/codes/radon_transforms/')
addpath('../ThirdParty/SeismicLab/codes/decon/')

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

figure; imagesc(offsets,tempo,dado);
figure; imagesc(offsets,tempo,dado_p);

% Calculo e aplicacao da divergencia esferica
div_esf = (1./(tempo*1500)).';
div_esf(1) = 1;

Div_esf = div_esf*ones(1,length(offsets));

dado_div = dado.*Div_esf;
dado_p_div = dado_p.*Div_esf;

figure(1) 
plot(tempo(1:400),dado(1:400,22)/max(dado(1:400,22)),'k');
title('Traço original')
grid

figure(2)
plot(tempo(1:400),dado_div(1:400,22)/max(dado_div(1:400,22)),'b');
title('Traço com compensação de divergência esférica')
grid
