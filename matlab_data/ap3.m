function [err,H,U,itr]=ap3(Uk,id,prt)
% Yuchen's trial for alternate projection U--> H
[n, k] = size(Uk);
H = sparse(n,k);
% cannot work well under crit2
if id==1
%Z = eye(k);
%Ni=[speye(k);sparse(1:n-k,ones(1,n-k),ones(1,n-k),n-k,k,n-k)];
Ni=[speye(k);sparse(n-k,k)];
Ni=Ni(randperm(n),:);
Ni=Ni(:,randperm(k));
Ni=normc(Ni);
[S,~,V]=svd(Uk'*Ni,0);
Z=S*V';
%load('Zo.mat','Z');
else 
[Z,R] = qr(rand(k));
Z=Z*diag(sign(diag(R)));
end

err_new = -1;
err = [];
itr = 0;
% alternate projection
while itr<3000
    U=Uk*Z;
    % adjust +- notation; M1: change nonpositive column; M2: change most
    % negative column; both make little influence
%     [~,idx] = max(U);
%     [~,idy] = max(abs(U));
%      id = find(idx~=idy)';
%      U(:,id) = -U(:,id); 
    H_old = H;
    [val,ind]= max(U,[],2);
    H = sparse(1:n,ind,val,n,k,n);
    err_old = err_new;
    err_new = norm(U-H,'fro');
    err = [err err_new];
    crit1 = norm(H-H_old,'fro')<1e-6;
    crit2 = abs(err_new-err_old)<1e-6;
    if crit1
        break;
    end
    [S,~,V]=svd(Uk'*H,0);
    Z = S*V';
    itr=itr+1;
end
% draw the results
if prt
close all
figure;
plot(err,'-r');
title('The iterative value of min||U-H||_F^2');
end