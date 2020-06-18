function [idx,member_id,outlier_id,err,data_err]=kind_ot_admm(Uk,mu,options)
%====================================================================
% Solving K-indicators by Alternating Projection algorithm with 
% Outlier-tolerant based on ADMM. (KindAP)
%
% Copyright: Yuchen Yang, Yin Zhang. 2018
% Reference: Stephen Boyd, Neal Parikh, Eric Chu Borja Peleato and Jonathan Eckstein
% Distributed Optimization and Statistical Learning via the 
% Alternating Direction Method of Multipliers.
% Last Modified: Yuchen Yang 05/27/2020
%====================================================================
% Input:
% Uk: n by k column-orthonormal matrix
%     (usually k eigenvectors of a Gram or Laplacian matrix)
% mu: the parameter of strength of outliers
%====================================================================
% Output:
% idx: cluster indices for data points
% itr: number of iterations
% err: the objective function value of outlier-tolerant K-indicators
%====================================================================
% The optimization problem is formulated as follows:
% argmin_{H,Z,V,W} 1/2||WZ-H||_F^2+mu||V||_{2,1}
% s.t. W,H,V in R^{n*k} (the same shape as give orthonormal matrix Uk) 
%      Z in R^{k*k}, Z^Z=I, H^H=I, H>=0
%      W-V-Uk=0, W^TW=I
%====================================================================
    if nargin < 3 || isempty(options), options = struct([]); end
    if isfield(options,'prt'), prt = options.prt; else, prt = 0; end
    if isfield(options,'maxitr'), max_itr=options.maxitr; else, max_itr=50; end 
    if isfield(options,'isnrmU'), isnrmrowU = options.isnrmU; else, isnrmrowU = 1; end
    if isfield(options,'isnrmH'), isnrmcolH = options.isnrmH; else, isnrmcolH = ~isnrmrowU; end
    if isfield(options,'maxit1'), maxit1 = options.maxit1; else, maxit1=2; end
    if isfield(options,'maxit2'), maxit2 = options.maxit2; else, maxit2=100;end
    if isfield(options,'Z'), Z = options.Z; else, Z=eye(size(Uk,2)); end
    if isfield(options,'e_abs'), e_abs = options.e_abs; else, e_abs = 1e-3; end
    if isfield(options,'e_rel'), e_rel = options.e_rel; else, e_rel = 1e-6; end
    if isfield(options,'update_rho'), update_rho = options.update_rho; else update_rho = 1; end
    % initialization  
    err = [];
    P = []; D = []; clust_err = []; data_err = [];
    kindap_options.maxit1 = maxit1;
    kindap_options.maxit2 = maxit2;
    kindap_options.isnrmU = isnrmrowU;
    kindap_options.isnrmH = isnrmcolH;
    kindap_options.prt = prt;
    if isnrmrowU
        Uk=normr(Uk);
    end
    [n,k]=size(Uk);
    V = zeros(n,k);
    W = Uk;
    W_old = zeros(n,k);
    L = zeros(n,k);
    H = zeros(n,k);
    rho = 10;
    for itr=1:max_itr
        % minimize over H,Z by KindAP
        H(max(abs(V),[],2)~=0,:) = 0;
        if isnrmcolH
            H=normc(H);
        end
        [s,~,v] = svd(W'*H);
        Z = s*v'; kindap_options.Z = Z;
        [index,H] = KindAP(W,k,kindap_options);
        if isnrmcolH
            H=normc(H);
        end
        [s,~,v] = svd(W'*H); Z = s*v';
        clust_err =  [clust_err; 0.5*norm(W-H*Z','fro')^2];
        B=W-Uk;
        % minimize V by soft thresholding
%         V=prox_l2(B+L/rho,2*mu/rho);
        [V,x]=prox_l2_adaptive(B+L/rho);
        % minimize W by Procrustes subproblem
        A=V+Uk;
%         [s,~,v]=svd((rho*A-L+H*Z')/(rho+1),0);
%         W=s*v';
        W=(rho*A-L+H*Z')/(1+rho);
        % update dual variable L
        L=L+rho*(W-A);
        S=-rho*(W-W_old);
        W_old=W;
        % save the objectives at each step
        obj = 0.5*norm(W-H*Z','fro')^2+x*sum(vecnorm(V,2,2))+trace(L'*(W-A))+rho/2*norm(W-A,'fro')^2;
        % obj = 0.5*norm(W-H*Z','fro')^2+mu*sum(vecnorm(V,2,2))+trace(L'*(W-A))+rho/2*norm(W-A,'fro')^2;
        err = [err;obj];
        % stopping criteria
        prim_res = norm(W-A,'fro');
        dual_res = norm(S,'fro');
        P = [P; prim_res];
        D = [D; dual_res];
        data_err = [data_err; norm(B,'fro')];
        if prt >= 1
            fprintf("Step: %2d, Primal Residual: %1.5e, Dual Residual: %1.5e, Obj: %1.5e\n", itr, prim_res,dual_res, obj);
        end
        prim_f = prim_res <= sqrt(n*k)*e_abs+max(k,norm(V,'fro'))*e_rel;
        dual_f = dual_res <= sqrt(n*k)*e_abs+norm(L,'fro')*e_rel;
        if prim_f && dual_f
            break;
        end
        % update rho
        if update_rho
            if prim_res > 10*dual_res
                rho=rho*2;
            end
            if prim_res < 0.1*dual_res
                rho=rho/2;
            end
        end
    end
    % get the outliers
    member_id = find(max(abs(V),[],2)==0);
    outlier_id = find(max(abs(V),[],2)~=0);
    idx = index;
    % plot
    if prt == 2
        figure;
        subplot(1,2,1);
        plot(err,'-r');
        hold on
        plot(clust_err,'--k');
        hold on 
        plot(data_err,'-.b')
        title('The Objective Value Curve','FontSize',20);
        h=legend('Lagrangian Error','Clustering Error','Data Change');
        set(h,'Fontsize',15);
        set(gca,'FontSize',13);
        
        subplot(1,2,2);
        semilogy(P,'-.b');
        hold on
        semilogy(D,'-g');
        title('Primal and Dual Residuals','FontSize',20);
        h=legend('Primal Residual','Dual Residual');
        set(h,'Fontsize',15);
        set(gca,'FontSize',13);
        
        save Kind_ot_admm_log.mat err clust_err data_err P D 
    end
    