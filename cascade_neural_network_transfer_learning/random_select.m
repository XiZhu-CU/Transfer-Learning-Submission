%% Function
function [x_tune,y_tune] = random_select(x_tune,y_tune,samplesize)
figure(999); set(gcf,'color','w','Position',[50,50,900,600]);
fig = gcf;
fig.PaperPositionMode = 'auto';
histogram(y_tune,15); hold on;
dice1 = randperm(length(y_tune));
x_tune = x_tune(dice1(1:samplesize),:);
y_tune = y_tune(dice1(1:samplesize));
histogram(y_tune,15); %legend('original','selected');
xlabel('Age (years)'); ylabel('Counts');
end