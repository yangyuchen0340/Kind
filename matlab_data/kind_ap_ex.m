function [ind,N,gres,exitflag,iter]=kind_ap_ex(M,k,max_epoch)
%====================================================================
% Solving K-indicators by Alternating Projection algorithm (KindAP)
% Extended version: 
% a. support inner projection to a sparser matrix than N
% b. support using higher-dimensional orthogonal feature space than k
% Copyright: Yuchen Yang. 2018
%====================================================================
% Input:
% M: n by d column-orthonormal matrix
%     (usually d eigenvectors of a Gram or Laplacian matrix)
% k: the number of clusters
% max_epoch: the number of epoches to train this model
%====================================================================
% Output:
% ind: n*1 vector representing cluster indices for data points
%   N: n by k indicator matrix representing the soft thresholding results
% iter: the number of total iterations
% gerr: the objective function value of outlier-tolerant K-indicators
% exitflag: how this algorithm is terminated, now 1 means a good stop.
%====================================================================
    % initialization
    [n,d]=size(M);
    [X0,~]=qr(randn(d,k),0);
    X=X0';
    tol=1e-5;
    gres=zeros(1,max_epoch);
    iter=0;
    % set threshold
    thres=0;
    % outer iterations
    for epoch=1:max_epoch
        res_old=-1;
        U=M*X';
        % flip the negative rows caused by projection-back procedure
        negative_rows=max(U,[],2)<0;
        U(negative_rows,:)=-U(negative_rows,:);
        % inner iteration
        for j=1:500
            % project onto N (extended by any threshold)
            N=proj_S(U,thres);
            % project back
            X=N'*M;
            [s,~,v]=svd(X,'econ');
            X=s*v';
            U=M*X';
            % calculate residual and stopping criteria
            res=norm(U-N,'fro');
            if j>2 && abs((res-res_old)/res_old)<tol
                break
            end
            res_old=res;
        end
        iter=iter+j;
        % update threshold (can be finetuned later)
        thres=thres+0.2*k/n;
        [val,ind]=max(N,[],2);

        % round to H
        H = sparse(1:n,ind,ones(n,1),n,k,n);
        column_sums = sum(H);
        if min(column_sums)==0
            H = H./min(sqrt(column_sums),1);
        else           
            H = H./sqrt(column_sums);
        end
        % projection back procedure
        X=H'*M;
        gres(epoch)=norm(M-H*X,'fro');
        if epoch>2 && abs((gres(epoch)-gres(epoch-1))/gres(epoch-1))<tol
            gres(epoch+1:end)=[];
            exitflag=1;
            break
        end   

        [S,~,V]=svd(X,'econ');
        X=S*V';
    end
    exitflag=0;
end


function S=proj_S(U,thres)
    % Projection with threshold
    if thres<0
        error('The input threshold should be greater than 0\n.');
    end
    % For all U_ij<thres, set U_ij=thres
    S=max(U,thres);
    % Retain a projection by setting u_ij<thres/2 to be 0
    if thres>0
        S(U<thres/2)=0;
    end
end