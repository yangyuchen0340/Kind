% perturbation stats
k=10;
%A=rand(10,256);
A=[eye(100) zeros(100,156)];
S=repmat(A,10,1);
rad=[];
gap=[];
vol=1e-3;
for j=1:100
    vol=vol*1.15;
    noise=rand(size(S))-0.5;
    SS=S+vol*normr(noise);
    [A,~,~]=svd(SS,'econ');
    A=A(:,1:k);
    g=apstat(A,100,0);
    %r=norm(SS-S,'fro');
    gap=[gap g];
    rad=[rad vol];
    if mod(j,10)==0
        fprintf('iteration = %d\n',j);
    end
end
loglog(rad,gap);
% plotyy(1:20,gap/log(10),'g',1:20,res/log(10),'r');
title('Log-Log Plot of radius-gap relationship')
% hold on
% plot(1:20,res,'r');