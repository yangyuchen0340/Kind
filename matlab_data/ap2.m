% Yuchen's second trial of alternating projection
function [gerr,idx,repeat,H,N]=ap2(Uk,id,prt)
    [n,k]=size(Uk);
    gerr=[];
    if id==1
        Zo = eye(k);
    else 
        [Zo,~] = qr(rand(k));    
    end

    % outer iteration H
    for repeat=1:k+1
        % internal iteration  (U,N) projection
        Z=Zo;
        % set parameters
        err_new = -1;
        % alternate projection
        for itr=1:5000
            U=Uk*Z;
            N = max(0,U);
            err_old = err_new;
            % can be further accelerated
            T = Uk'*N;
            [S,D,V]=svds(T,k,'largest','Tolerance',1e-10);
            Z = S*V';
        %    err_new = norm(U-N,'fro');
            err_new = norm(diag(D)-1);
        %    err = [err err_new];
            if abs(err_old-err_new)<=1e-6
                break;
            end

        end
        % N project onto H
        % maximum element in each row
        [val,ind]= max(N,[],2);
        H = sparse(1:n,ind,val,n,k,n);

        % normalization is not necessary
        % H = normc(H);

        % maximum element in each column
        % N=normr(N);
        % [val,ind]= max(N);
        % H = sparse(ind,1:k,val,n,k,k);
        % H = normc(H);

        res = norm(U-H,'fro');
        % This stopping criteria needs to be polished.
        if repeat>1&&(res>gerr(end)+1e-4 || abs(res-gerr(end))/gerr(end)<1e-8)
            break;
        end
        gerr = [gerr res];
        % H project back onto U;
        [S,~,V]=svd(Uk'*H,0);
        Zo=S*V';
        % some processessing
        % Zo =makeblk(H,Uk,Zo,1);
    end

    [val,ind]= max(N,[],2);
    H = sparse(1:n,ind,val,n,k,n);
    % get the clustering
    [i,j]=find(H);
    idx(i)=j;

    % draw the results
    if prt
        figure;
        plot(gerr,'-r');
        title('The iterative value of min||U-H||_F^2');
    end

end