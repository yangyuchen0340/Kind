function rate = purity(predicted_labels,real_labels)

M = crosstab(predicted_labels,real_labels); % you can use also use "confusionmat"
nc = sum(M,1);
mc = max(M,[],1);
rate = sum(mc(nc>0))/sum(nc);

end
