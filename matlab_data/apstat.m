% local minimum stats
function result=apstat(A,maxit,prt)
% different trial for random matrices and special synethic data
% function gap=apstat(a,b,prt)

% maxit=100;
data=zeros(maxit,1);
parfor sp=1:maxit
%     [err,N]=ap3(A,1,0);
%     data(sp)=length(find(max(N)<0.2));
      [~,~,dis] = kmeans(A,size(A,2),'Start','plus','Replicates',10);
      data(sp) = norm(dis);
%       err=ap3(A,0,0);
%       data(sp)=err(end);
end
if prt
figure;
plot(data,'*r','Markersize',20);
axis([0,100,-0.2,6])
grid on
set(gca,'Fontsize',20)
% hold on
% err=ap1(A,1,0);
% plot(floor(maxit/2),err(end),'*');
title('Objectives of K-means++','Fontsize',25);
end
result=data;
end