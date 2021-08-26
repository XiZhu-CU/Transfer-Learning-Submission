 %% function setaf2
function af = setaf2(str)
if strcmp(str,'Random') == 1
    n1 = randperm(4);
    if n1(1) == 1
        af = 'logsig';
    elseif n1(1) == 2
        af = 'tansig';
    elseif n1(1) == 3
        af = 'satlin';
    elseif n1(1) == 4
        af = 'poslin';
    end
else
    af = 'logsig';
end
end
