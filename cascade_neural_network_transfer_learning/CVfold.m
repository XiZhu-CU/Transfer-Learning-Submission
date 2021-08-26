function array = CVfold(input,resp,fold)
numo = size(input,1); % number of observations
foldsize = floor(numo/fold);
list = randperm(numo);
array = {};
for i=1:fold
    temp_list = list;
    val_cor = temp_list((i-1)*foldsize+1:i*foldsize);
    temp_list((i-1)*foldsize+1:i*foldsize) = [];
    tr_cor = temp_list;
    array{1,i} = input(tr_cor,:); % training set
    array{2,i} = input(val_cor,:); % validation set
    array{3,i} = resp(tr_cor); % training set- response
    array{4,i} = resp(val_cor); % validation set- response
    if i == fold
        temp_list = list;
        val_cor = temp_list((i-1)*foldsize+1:numo);
        temp_list((i-1)*foldsize+1:numo) = [];
        tr_cor = temp_list;
        array{1,i} = input(tr_cor,:); % training set
        array{2,i} = input(val_cor,:); % validation set
        array{3,i} = resp(tr_cor); % training set- response
        array{4,i} = resp(val_cor); % validation set- response
    end
end
end
 