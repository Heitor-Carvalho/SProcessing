function [file_name_ext, xlabel_txt] = net_analisys_text(test_number)

  
  switch test_number
    case 1
      xlabel_txt = 'Faixa de pesos iniciais'
      file_name_ext = 'initial_weigth'
    case 2
      xlabel_txt = 'Número de neurônios'
      file_name_ext = 'mid_layer_sz'
    case 3
      xlabel_txt = 'Passo de predição'
      file_name_ext = 'prediction_step'
    case 4
      xlabel_txt = 'Compriment do filtro'
      file_name_ext = 'filter_length'
    case 5 
      xlabel_txt = 'Regularização'
      file_name_ext = 'regularization'
    otherwise
      xlabel_txt = ''
      file_name_ext = ''
  end

end
