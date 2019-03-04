close all
load result_DSC_data_2.mat
max_trial = size(AC,1)/4;
row_SR = 3; row_KI = 4; row_KL = 5; row_KM = 8;
row_SR1 = 11; row_KM1 = 12; 
for ds = 1:4
    currAC = AC(1+(ds-1)*max_trial:max_trial*ds,:);
    meanAC = mean(currAC);
    stdAC = std(currAC);
    maxAC = max(currAC);
    minAC = min(currAC);
    figure
    errorbar(0.9:1:5.9,meanAC([row_SR1,row_SR,row_KI,row_KL,row_KM1,row_KM]),...
        stdAC([row_SR1,row_SR,row_KI,row_KL,row_KM1,row_KM]),'or','capsize',15)
    hold on
    errorbar(1.1:1:6.1,meanAC([row_SR1,row_SR,row_KI,row_KL,row_KM1,row_KM]),...
        meanAC([row_SR1,row_SR,row_KI,row_KL,row_KM1,row_KM])-minAC([row_SR1,row_SR,row_KI,row_KL,row_KM1,row_KM]),...
        maxAC([row_SR1,row_SR,row_KI,row_KL,row_KM1,row_KM])-meanAC([row_SR1,row_SR,row_KI,row_KL,row_KM1,row_KM]),'*')
    xlabel('Algorithms: (1-2):SR 1 and 10, (3):KindAP, (4):KindAP+L, (5-6):K-means 1 and 10')
    ylabel('Accuracy')
    legend('std','max-min','location','best')
    set(gca,'fontsize',15)
end