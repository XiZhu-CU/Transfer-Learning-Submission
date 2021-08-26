%% manually Function
function cuscolormap = model_inference(hat_y,true_y,k)
cuscolormap = [linspace(0,0.2,128)',linspace(0,1,128)',linspace(1,0.2,128)';...
    linspace(0.2,0,128)',linspace(1,0,128)',linspace(0.2,1,128)'];
fprintf('Rho: %g, RMSE: %g, MAE: %g \n',...
    corr(hat_y,true_y),sqrt(mean((true_y-hat_y).^2)),mean(abs(true_y-hat_y)));
 
if nargin == 3
    figure(k); set(gcf,'color','w','Position',[200,200,900,600]);
    fig = gcf;
    fig.PaperPositionMode = 'auto';
    plot(linspace(-3,3,100),linspace(-3,3,100),'color',[0.8,0.8,0.8],...
        'linewidth',6,'LineStyle','-.'); hold on;
    tPAD = hat_y-true_y;
    scatter(true_y,hat_y,155,tPAD,'filled','MarkerEdgeColor',[0,0,0]);
    colorbar; colormap(cuscolormap)
    xlim([-3,3]); ylim([-3,3])
    xlabel('memory'); ylabel('Predicted memory');
    title(sprintf('PAD score: %g',mean(tPAD)));
    caxis([-3,3]);
    set(gca,'fontsize',14,'fontweight','bold'); grid on;
    set(gca,'color',[0.95,0.95,0.95]);
    [r,p]=corr(true_y,tPAD);
    fprintf('mem-related bias: %g, with p-value %g \n',r,p)
    fprintf('Mean PAD: %g ... \n',mean(tPAD));
end
end