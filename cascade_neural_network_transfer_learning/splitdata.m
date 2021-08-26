
function [dataTest,dataTrain] = splitdata(dataset,test_prop, seed)

    rng(seed);
    n=size(dataset,1);
    Indices=randperm(n);
    n_test=ceil(n*test_prop);
    dataTest=dataset(Indices(1:n_test),:);
    dataTrain=dataset(Indices((n_test+1):end),:);
end
