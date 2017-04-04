%%% Script para criar um cmp sintetico simulando multiplas de curto periodo
% Rafael Ferrari 06/08/2014

% eventos 1500m/s
%tau_1500=0.5*[0.2000    0.4000    0.6000    0.8000    1.0000    1.2000    1.4000 1.6 1.8 2.0 ].';
%amp_1500=[1.0000   -0.5000    0.2500   -0.1250    0.0625   -0.0312    0.0156 -(0.5^7) (0.5^8) -(0.5^9)].';

N_hip=25;
coef_ref=0.6;
t0_fundo_mar=0.1;
tau_1500=t0_fundo_mar*(1:1:N_hip).';
amp_1500=((-coef_ref).^(0:1:N_hip-1)).';
vel_1500=1500*ones(length(tau_1500),1);

%div_esf=1./(tau_1500*1500);
%amp_1500=div_esf.*amp_1500;

% Frequencia central da wavelet
f=30;

% Vetor de offsets em metros
h=0:5:3500;

% Configuracao de offsets cmp 1501 Jequitinhonha
%h=175:50:3125;

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

figure; plot(tempo,dado(:,1)/max(dado(:,1)),'k');
hold on
plot(tempo,dado_div(:,1)/max(dado_div(:,1)),'b');

figure; imagesc(offsets,tempo,dado_div);
figure; imagesc(offsets,tempo,dado_p_div);

[ns,nt]=size(dado);

WriteSegy('multiplas_5m.segy',dado,'dt',dt,'ns',ns,'offset',h,'cdp',1);
WriteSegy('primarias_5m.segy',dado_p,'dt',dt,'ns',ns,'offset',h,'cdp',1);

WriteSegy('multiplas_5m_div.segy',dado_div,'dt',dt,'ns',ns,'offset',h,'cdp',1);
WriteSegy('primarias_5m_div.segy',dado_p_div,'dt',dt,'ns',ns,'offset',h,'cdp',1);
