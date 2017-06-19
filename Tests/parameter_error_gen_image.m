% Script used to generate the MSE and MSE_p (MSE based on the primary trace)
% curve. As input espects the following variables present in the workset
% - MSE:          Mean square error vector
% - MSE_p:        Mean square error vector using the trace with only primary 
% - xlabel_txt:   Graph xlabel text
% - ylabel_txt:   Graph ylabel text
% - sweep_param:  Parameter changed vector

figure(1)
plot(sweep_param, mse)
xlabel(xlabel_txt);
ylabel(ylabel_txt);
title('Prediction MSE')
grid
saveas(gcf, sprintf('prediction_mse_changepar_%d.jgp', xlabel_txt));
savefig(sprintf('prediction_mse_changepar_%d.jgp', xlabel_txt));

figure(2)
plot(sweep_param, mse_p)
xlabel(xlabel_txt);
ylabel(ylabel_txt);
title('Only primary trace error')
grid
saveas(gcf, sprintf('primary_mse_changepar_%d.jgp', xlabel_txt));
savefig(sprintf('primary_mse_changepar_%d.jgp', xlabel_txt));



figure(3)
plot(sweep_param, mse_p)
xlabel(xlabel_txt);
ylabel(ylabel_txt);
title('Prediction error. Only primary trace error')
grid
saveas(gcf, sprintf('prediction_primary_mse_changepar_%d.jgp', xlabel_txt));
savefig(sprintf('prediction_primary_mse_changepar_%d.jgp', xlabel_txt));
