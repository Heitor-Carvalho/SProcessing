function [w,e,f,f1,f2]=double_gate_pef(trace,L,N1,N2,reg_fac)
% 
% [w,e,f,f1,f2]=double_gate_pef(trace,L,N1,N2,reg_fac)
%
% Double gate prediction error filter
%
% e(k) = trace(k) - f1*trace(k-L) - f2*trace(k-2L)
%
% Inputs:
%
% trace: time series (Nix1 vector)
% L: prediction lag
% N1: length of f1 (first gate)
% N2: length of f2 (second gate)
% reg_fac: regularization factor
%
% Outputs:
% 
% w: pef coeficients [1 0 ... 0 -f1 0 ... 0 -f2]
% e: estimated multiples
% f: estimated primaries
% f1: first gate coeficients
% f2: second gate coeficients
%
% Rafael Ferrari 09/02/2015

Ni = length(trace);

tmp = convmtx(trace,N2+L);

C = [tmp(:,1:N1) tmp(:,end-N2+1:end)];

clear tmp;

R = (C'*C);

[Nie tl] = size(C);

d = [trace(L+1:end);zeros(Nie-Ni+L,1)];

p = C'*d;

wo = (R + reg_fac*R(1,1)*eye(N1+N2))\p;

f1 = wo(1:N1);

f2 = wo(N1+1:end);

w = [1;zeros(L-1,1);-f1;zeros(L-N1,1);-f2];
whos w

f = conv(w,trace);

f = f(1:Ni);

%f=f(filter_delay+1:length(reference_signal)+filter_delay);

e = trace - f;

return