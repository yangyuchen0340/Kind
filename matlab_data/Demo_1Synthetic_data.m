%% Test both K-points and K-rays

clear
rng('default')

diary('result_AP_KI_KM.txt');
diary on;

run_kmeans = 1; No_kmeans = 1; Phase_kmeans = 'off';
correction = 0; Normalize = 0;
ktype = 0; % k-points = 0.  k-rays = 1 
rho = 1;

dim = 300; ndk = 100; delta = .99;
no_clusters = (10:30:160);
no_trials = 10;

fprintf('--------------------------------------------------------\n')
fprintf('Synthetic Data: [n,d,k] = [100k,300,k]; radius = 0.99\n') 
fprintf('--------------------------------------------------------\n')

ACa = zeros(length(no_clusters),no_trials); ACm = ACa; ACi = ACa;
T_ap = ACa; T_kmeans = ACa; T_kind = ACa;

for k_rep = 1:length(no_clusters)
        %% Data generation
        k = no_clusters(k_rep);
        n = ndk*k;
        
        
        for it_rep = 1: no_trials
        % Centers and Weights
        C = randn(k,dim); C = orth(C')'*sqrt(2);
        W = abs(randn(ndk,k));
        idxg=reshape(repmat(1:k,n/k,1),n,1); % ground truth index.
        W = W*ktype + rho;
        
        % data
        M = [];
        for i = 1:k
            M = [M ; W(:,i)*C(i,:)]; %#ok<AGROW>
        end
        N = randn(size(M));
        ss = sqrt(sum(M.^2,2))./sqrt(sum(N.^2,2));
        M = M + delta*bsxfun(@times,ss,N);
        
        if Normalize == 1
            Norms = sqrt(sum(M'.^2,1));
            M = (bsxfun(@times, M', 1./Norms))';
        end
        
        %% AP algorithm
        
%         s = zeros(n);
%         for i = 1:n
%             for j = 1:i
%                 s(i,j) = -sum((M(i,:)-M(j,:)).^2);
%             end
%         end
%         s = s + s';
%         p = 3*median(s)';
%         
%         tic; [idxa,netsim,dpsim,expref] = apcluster(s,p); t_ap = toc;
%         
%         ind = unique(idxa); idxa1 = idxa;
%         for i = 1: length(ind)
%             b = find(idxa == ind(i)); idxa1(b) = i*ones(length(b),1);
%         end        
        
        %%  SVD        
        tic, [Ut,St,Vt] = svd(M); U = Ut(:,1:k); t_svd = toc;
        
        %% KindAP
        options.rel = 0; options.idxg = idxg; options.it = 1; options.disp = 0;
        tic, [idx,C,f,out] = KindAP(U,k,options); t_kind = toc;  
        option.do_inner = 0;
        tic, idxa1 = KindAP(U,k,options); t_a = toc; % added Yuchen SR
        %% Kmeans
        tic, [idxm,~,sumD] = kmeans...
            (U,k,'Replicates',No_kmeans,'OnlinePhase',Phase_kmeans); t_kmeans = toc;
        fm = sum(sumD);
                
        %% Accuracy 
       
%         if length(ind) == k
            idxa1 = bestMap(idxg,idxa1);
            ACa(k_rep,it_rep) = 100*sum(idxg==idxa1)/n;
%         end
        
        idxm = bestMap(idxg,idxm);
        ACm(k_rep,it_rep) = 100*sum(idxg==idxm)/n;
        
        idx = bestMap(idxg,idx);
        ACi(k_rep,it_rep) = 100*sum(idxg==idx)/n;
        
       %% Running time
%         T_ap(k_rep,it_rep) = t_ap;
        T_ap(k_rep,it_rep) = t_svd + t_a;
        T_kind(k_rep,it_rep) = t_svd + t_kind;
        T_kmeans(k_rep,it_rep) = t_svd + t_kmeans;
        end
        
        %% Output information
      
        fprintf('******************************************************\n')
        fprintf('k = %3i; ACa = %6.2f%% ta = %3.2fs\n',k,mean(ACa(k_rep,:)),mean(T_ap(k_rep,:)))
        fprintf('         ACi = %6.2f%% ti = %3.2fs\n',mean(ACi(k_rep,:)),mean(T_kind(k_rep,:))),
        fprintf('         ACm = %6.2f%% tm = %3.2fs\n',mean(ACm(k_rep,:)),mean(T_kmeans(k_rep,:))),
        
end

diary off;
%% Plot figures

    figure(1)
    subplot(121)
    axis equal
    h = plot(no_clusters,mean(ACi,2),'ro-',no_clusters,mean(ACa,2),'b+-',no_clusters,mean(ACm,2),'gd-');
    set (gca,'fontsize',15,'XTick',no_clusters,'xlim',[no_clusters(1),no_clusters(end)])
    hh = legend('\color{red} KindAP','\color{blue} AP','\color{green} K-means','location','SouthWest');
    set(hh,'FontSize',20);
    title('Data size: [n,d,k] = [100k,300,k]','FontSize',20)
    xlabel('No. of Clusters','FontSize',20)
    ylabel('Clustering Accuracy','FontSize',20)
    set(h,'linewidth',3,'MarkerSize',10); grid on; drawnow, shg
    
    subplot(122)
    axis equal
    h = semilogy(no_clusters,mean(T_kind,2),'ro-',no_clusters,mean(T_ap,2),'b+-',no_clusters,mean(T_kmeans,2),'gd-');
    set (gca,'fontsize',15,'XTick',no_clusters,'xlim',[no_clusters(1),no_clusters(end)])
    hh = legend('\color{red} KindAP','\color{blue} AP','\color{green} K-means','location','SouthWest');
    set(hh,'FontSize',20);
    title('Data size: [n,d,k] = [100k,300,k]','FontSize',20)
    xlabel('No. of Clusters','FontSize',20)
    ylabel('Running Time: log(T)','FontSize',20)
    set(h,'linewidth',3,'MarkerSize',10); grid on; drawnow, shg
    
    
    save synthetic_result.mat no_clusters no_trials No_kmeans dim n k ACa ACi ACm T_ap T_kind T_kmeans ktype rho
    