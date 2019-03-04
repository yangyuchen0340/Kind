close all
load result_DSC_data_2.mat
max_trial = size(AC,1)/4;
row_SR = 3; row_KI = 4; row_KL = 5; row_KM = 8;
for ds = 1:4
    currAC_SR = AC(1+(ds-1)*max_trial:max_trial*ds,row_SR);
    currAC_KI = AC(1+(ds-1)*max_trial:max_trial*ds,row_KI);
    currAC_KL = AC(1+(ds-1)*max_trial:max_trial*ds,row_KL);
    currAC_KM = AC(1+(ds-1)*max_trial:max_trial*ds,row_KM);
    min_thres = 0.9*min([min(currAC_SR),min(currAC_KL),min(currAC_KM),min(currAC_KI)]);
    max_thres = min(100,1.1*max([max(currAC_SR),max(currAC_KL),max(currAC_KM),max(currAC_KI)]));
    figure
    histogram(currAC_SR,min_thres:.5:max_thres,'facealpha',.5,'edgecolor','none')
    hold on
    histogram(currAC_KI,min_thres:.5:max_thres,'facealpha',.5,'edgecolor','none')
    hold on
    histogram(currAC_KL,min_thres:.5:max_thres,'facealpha',.5,'edgecolor','none')
    hold on
    histogram(currAC_KM,min_thres:.5:max_thres,'facealpha',.5,'edgecolor','none')
    legend('SR 10','KindAP','KindAP+L','K-means 10')
    xlabel('Accuracy')
    ylabel('Frequency Distribution')
    set(gca,'fontsize',15)
end