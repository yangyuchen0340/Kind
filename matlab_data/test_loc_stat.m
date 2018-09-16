% local minimum stats
function result=test_loc_stat(A,maxit,prt)
% different initial trials for different local minima
    data=zeros(maxit,1);
    parfor sp=1:maxit
          [~,~,dis] = kmeans(A,size(A,2),'Start','plus','Replicates',10);
          data(sp) = norm(dis);
%           KindAP
%           err=kind_ap(A,0,0);
%           data(sp)=err(end);
    end
    % plot
    if prt
        figure;
        plot(data,'*r','Markersize',15);
        axis([0,100,-0.2,6])
        grid on
        set(gca,'Fontsize',20)
        title('Objectives of K-means++','Fontsize',25);
    end
    result=data;
end