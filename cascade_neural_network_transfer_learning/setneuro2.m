%% Function setneuro2
 
function numN = setneuro2(numL,dim_in)
numN = zeros(1,numL);
sup = floor(log2(dim_in));
numN = 2.^randperm(sup,numL);
numN = sort(numN,'descend');
end
