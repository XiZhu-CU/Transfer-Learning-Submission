function pt_model = freezelayer(pt_model,fot)
% freeze layers in the model
% check the consistency b/w pt_model and fot
if pt_model.numLayers == length(fot)
    if fot(1) == true
        for ly = 1:numel(pt_model.inputWeights)
            pt_model.inputWeights{ly}.learn = false;
        end
        pt_model.biases{1}.learn = false;
    else
        for ly = 1:numel(pt_model.inputWeights)
            pt_model.inputWeights{ly}.learn = true;
        end
        pt_model.biases{1}.learn = true;
    end
    
    for arg = 2:length(fot)
        if fot(arg) == true
            for ly = 1:numel(pt_model.layerWeights(arg,:))
                try
                    pt_model.layerWeights{arg,ly}.learn = false;
                end
            end
            pt_model.biases{arg}.learn = false;
        else
            for ly = 1:numel(pt_model.layerWeights(arg,:))
                try
                    pt_model.layerWeights{arg,ly}.learn = true;
                end
            end
            pt_model.biases{arg}.learn = true;
        end
    end
else
    error('The dimension of the freezable vector is not consistent with the number of layers in the model!');
end
end