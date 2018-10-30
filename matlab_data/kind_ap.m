function [gerr,idx,repeat,H,Z]=kind_ap(Uk,id,prt)
%====================================================================
% Solving K-indicators by Alternating Projection algorithm (KindAP)
%
% Copyright: Yuchen Yang. 2016
%
%====================================================================
% Input:
% Uk: n by k column-orthonormal matrix
%     (usually k eigenvectors of a Gram or Laplacian matrix)
% id: type of initial guess: 0 for random and 1 for identity
% prt: whether printing the residual curve or not
%====================================================================
% Output:
% idx: n*1 vector representing cluster indices for data points
%   H: n by k indicator matrix
% repeat: number of outer iterations
% gerr: the objective function value of outlier-tolerant K-indicators
%====================================================================
    % initialization
    [n,k]=size(Uk);
    % maximum inner iteration numbers
    max_in=5000;
    gerr=[];
    if id==1
        Zo = eye(k);
    else 
        [Zo,~] = qr(rand(k));    
    end

    % outer iteration (U,H) projection
    for repeat=1:k+1
        Z=Zo;
        err_new = -1;
        % internal iteration (U,N) alternating projection
        for itr=1:max_in
            U=Uk*Z;
            % project U to N
            N = max(0,U);
            err_old = err_new;
            % project N to U
            % might be further accelerated
            T = Uk'*N;
            [S,D,V]=svds(T,k,'largest','Tolerance',1e-10);        
            Z = S*V';
            % stopping criteria: err_new = norm(U-N,'fro');
            err_new = norm(diag(D)-1);
            if abs(err_old-err_new)<=1e-6
                break;
            end

        end
        % N project onto H
        % maximum element in each row
        [val,ind]= max(N,[],2);
        H = sparse(1:n,ind,ones(n,1),n,k,n);

        % normalization is not necessary
        % normc will harm zero columns
        % H = normc(H);

        % another option: maximum element in each column
        % N=normr(N);
        % [val,ind]= max(N);
        % H = sparse(ind,1:k,val,n,k,k);
        % H = normc(H);

        res = norm(U-H,'fro');
        % This stopping criteria can be polished.
        if repeat>1&&(res>gerr(end)+1e-4 || abs(res-gerr(end))/gerr(end)<1e-8)
            break;
        end
        gerr = [gerr res];
        % H project back onto U;
        [S,~,V]=svd(Uk'*H,0);
        Zo=S*V';
        % generate block diagonal if necessary when rank deficiency happens
        % Zo =makeblk(H,Uk,Zo,1);
    end

    [val,ind]= max(N,[],2);
    % can also use H = sparse(1:n,ind,val,n,k,n);
    H = sparse(1:n,ind,ones(n,1),n,k,n);
    % get the clustering indices
    idx = ind;

    % draw the results
    if prt
        figure;
        plot(gerr,'-r');
        title('The iterative value of min||U-H||_F^2');
    end

end
