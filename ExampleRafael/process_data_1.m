clear all 
close all

% Open and read files of synthetic primaries and multiples
prim = loadsegy('primarias_5m.segy');
mult = loadsegy('multiplas_5m_div.segy');

data_prim = double(prim.tr_data); 
data_mult = double(mult.tr_data);

dt = 0.004;                           % temporal sampling in seconds 
dx = 5;                               % spatial sampling in meters

[Nsample,NRic] = size(data_mult);
time = (0:dt:dt*(Nsample-1));
h = [0:dx:(NRic-1)*dx];

%figure; plotseis(data_prim,time,h,1,[5 max(abs(data_prim(:)))],1,1,'k');
%figure; plotseis(data_mult,time,h,1,[5 max(abs(data_prim(:)))],1,1,'k');

% Apply smoothing window to the dataset
data_box = ones(size(data_prim)); 
ll = 10;
wind = hanning(2*ll+1);
data_box(:,end-ll:end) = ones(size(data_box,1),1)*(wind(end-ll:end).');
%figure; imagesc(data_box);

data_prim_smooth = data_prim.*data_box; 
data_mult_smooth = data_mult.*data_box;
%figure; plotseis(data_prim_smooth,time,h,1,[5 max(abs(data_prim(:)))],1,1,'k');
%figure; plotseis(data_mult_smooth,time,h,1,[5 max(abs(data_prim(:)))],1,1,'k');

% Compute the tau-p transform of the dataset using the code of Sacchi
% q = (0:10^(-6):10^(-3));
%q = (-10^(-3):10^(-5):10^(-3));
h_fo150 = h(31:end);
q = linspace(0,7e-4,length(h));
%q_fo150 = linspace(0,7e-4,length(h_fo150));
q_fo150 = linspace(0,7e-4,length(h));
%q = linspace(0,7e-4,length(h));
N = 1; 
flow = 3; 
fhigh = 80; 
mu = .010;
sol = 'ls';

radon_prim = inverse_radon_freq(data_prim,dt,h,q,N,flow,fhigh,mu,sol);
radon_mult = inverse_radon_freq(data_mult,dt,h,q,N,flow,fhigh,mu,sol);
radon_mult_fo150 = inverse_radon_freq(data_mult(:,31:end),dt,h_fo150,q_fo150,N,flow,fhigh,mu,sol);

% figure; imagesc(q, time, radon_prim); caxis([-0.003, 0.003]);
figure; imagesc(q, time, radon_mult); caxis([-1e-5, 1e-5]);
figure; imagesc(q, time, radon_mult_fo150); caxis([-1e-5, 1e-5]);

% Compute the tau-p transform of the dataset in the time-offset domain
% mapping = [0:Nsample:Nsample*(NRic-1)];
% lineT = zeros(size(mapping));
% radon_mult_v2 = zeros(size(radon_mult));
% 
% tic
% for ik=1:Nsample 
%     for jk = 1:length(q)
%         lineT = round((time(ik) + q(jk)*h)/dt);  %calculate the points of linear events
%         good1 = find((lineT>0) & (lineT<Nsample+1));
%         line = lineT + mapping;
%         radon_mult_v2(ik,jk) = sum(data_mult_smooth(line(good1)));  % Radon map
%     end
% end
% toc
% figure; imagesc(q,time, radon_mult_v2); caxis([-5,5]);

pn = 20;

% figure,plot(0:dt:dt*(Nsample-1),radon_mult(:,pn)/max(abs(radon_mult(:,pn))))
% hold on
% plot(0:dt:dt*(Nsample-1),radon_mult_fo150(:,pn)/max(abs(radon_mult_fo150(:,pn))),'k')
% 
% figure,plot(0:dt:dt*(Nsample-1),radon_mult(:,pn))
% hold on
% plot(0:dt:dt*(Nsample-1),radon_mult_fo150(:,pn),'k')

acor = xcorr(radon_mult(:,pn),radon_mult(:,pn),'coef');
acor = acor(((length(acor)-1)/2)+1:end);
% figure, plot(acor);

predictorLength = 70;
predictionLag = 10;

[filtro,dec_res] = predictive(radon_mult(:,pn),predictorLength,predictionLag,0.1);
% figure,plot(0:dt:dt*(Nsample-1),dec_res); 28

acor_fo150 = xcorr(radon_mult_fo150(:,pn),radon_mult_fo150(:,pn),'coef');
acor_fo150 = acor_fo150(((length(acor_fo150)-1)/2)+1:end);
% figure, plot(acor_fo150,'k');
% hold on, plot(acor)

[filtro_fo150,dec_res_fo150] = predictive(radon_mult_fo150(:,pn),predictorLength,predictionLag,0.1);
% figure,plot(0:dt:dt*(Nsample-1),dec_res_fo150);

Tsamples = 626;
% Influencia da ausencia dos afastamentos iniciais
figure,subplot(2,2,1),plot(0:dt:dt*(Tsamples-1),radon_mult(1:Tsamples,pn),'k','LineWidth',2)
xlabel('Tau (s)')
title('Tra�o no dominio transformado (afastamento inicial de 0m)','FontSize',14,'FontWeight','bold')
subplot(2,2,3),plot(0:dt:dt*(Tsamples-1),dec_res(1:Tsamples),'k','LineWidth',2)
xlabel('Tau (s)')
title('Resultado da filtragem preditiva (afastamento inicial de 0m)','FontSize',14,'FontWeight','bold')
subplot(2,2,2),plot(0:dt:dt*(Tsamples-1),radon_mult_fo150(1:Tsamples,pn),'k','LineWidth',2)
xlabel('Tau (s)')
title('Tra�o no dominio transformado (afastamento inicial de 150m)','FontSize',14,'FontWeight','bold')
subplot(2,2,4),plot(0:dt:dt*(Tsamples-1),dec_res_fo150(1:Tsamples),'k','LineWidth',2)
xlabel('Tau (s)')
title('Resultado da filtragem preditiva (afastamento inicial de 150m)','FontSize',14,'FontWeight','bold')

%hold on,plot(radon_mult_v2(:,10)/max(radon_mult_v2(:,10)),'k')
%figure,plot(radon_mult(:,1)/max(radon_mult(:,1)))

%figure,imagesc(abs(fftshift(fft2(data_mult)))) % fk dado suavizado

return

radon_dec = zeros(size(radon_mult));
pred_length = 28;
lag0 = 14;
dp = q(2)-q(1);
lag = zeros(length(q),1);

for ip=1:length(q)
    
    if (1-(1500*(ip-1)*dp)^2) < 0,
        lag(ip) = 1;
    else
         lag(ip) = floor(lag0*dt*sqrt(1-(1500*(ip-1)*dp)^2)/dt);
         if lag(ip) < 1; lag(ip) = 1; end
    end
    
    [filtro,dec_res] = predictive(radon_mult(:,ip),28,lag(ip),0.1);
    radon_dec(:,ip) = dec_res;
end

figure,plot(radon_mult(:,1))
%hold on
%plot(dec_res,'k')

file = fopen('radon_mult.bin','w+');
fwrite(file,radon_mult,'float');
fclose(file);

system('suaddhead ns=751 <radon_mult.bin | sushw key=dt a=4000 | sushw key=cdp a=1 >radon_mult.su')
system('suximage <radon_mult.su perc=100 windowtitle="Radon Freq" &')


file = fopen('radon_dec.bin','w+');
fwrite(file,radon_dec,'float');
fclose(file);

system('suaddhead ns=751 <radon_dec.bin | sushw key=dt a=4000 | sushw key=cdp a=1 >radon_dec.su')
system('suximage <radon_dec.su perc=100 windowtitle="Radon Freq Dec" &')


prim_est = forward_radon_freq(radon_dec,dt,h,q,1,flow,fhigh);

file = fopen('prim_est.bin','w+');
fwrite(file,prim_est,'float');
fclose(file);

system('suaddhead ns=751 <prim_est.bin | sushw key=dt a=4000 | sushw key=cdp a=1 >prim_est.su')
system('suximage <prim_est.su perc=100 windowtitle="Primarias estimadas" &')

file = fopen('prim.bin','w+');
fwrite(file,data_prim,'float');
fclose(file);

system('suaddhead ns=751 <prim.bin | sushw key=dt a=4000 | sushw key=cdp a=1 >prim.su')
system('suximage <prim.su perc=100 windowtitle="Primarias" &')
