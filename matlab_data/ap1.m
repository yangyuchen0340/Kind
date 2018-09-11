function [err,N,U,itr]=ap1(Uk,id,prt)
% Yuchen first trial for alternate projection
% set parameters
% [S,~,~] = svd(A,0);
[n, k] = size(Uk);
N = zeros(n,k);
% Uk=S(:,1:k);
% set initial guess
if id==1
Z = -eye(k);

% Ni=[speye(k);sparse(n-k,k)];
% Ni=Ni(randperm(n),:);
% Ni=Ni(:,randperm(k));
% Ni=normc(Ni);
% [S,~,V]=svd(Uk'*Ni,0);
% Z=S*V';
else 
[Z,R] = qr(rand(k)); 
Z=Z*diag(sign(diag(R)));
end

err_new = -1;
err = [];
itr = 0;
% alternate projection
while itr<500
    U=Uk*Z;
    % adjust +- notation; M1: change nonpositive column; M2: change most
    % negative column; both make little influence
%     [~,idx] = max(U);
%     [~,idy] = max(abs(U));
%      id = find(idx~=idy)';
%      U(:,id) = -U(:,id); 
    N_old = N;
    N = max(0,U);
    err_old = err_new;
    err_new = norm(U-N,'fro');
    err = [err err_new];
    crit1 = norm(N-N_old,'fro')<1e-4;
    crit2 = abs(err_new-err_old)<1e-4;
    if crit2
        break;
    end
    [S,D,V]=svd(Uk'*N,0);
%     rk=size(D,1);
    % newly add 11/14/17
%     rkind=find(diag(D)>1e-7);
%     if ~isempty(rkind)
%        rk=rkind(end);
%     end
%     Z = S(:,1:rk)*V(1:rk,:);
    Z=S*V';
    itr=itr+1;
end
% draw the results
if prt
close all
figure;
plot(err,'-r');
title('The iterative value of min||U-N||_F^2');
end