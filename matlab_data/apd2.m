k=5;
d=10;
m=2;
A=[eye(k) zeros(k,d-k)];
lam=rand(m,2);
M=kron(A,lam);
addnoise=0;

if addnoise
noise=rand(size(M))-0.5;
M=M+addnoise*normr(noise);
end

[s,~,~]=svd(M,'econ');
Uk=s(:,1:2*k);
 
%[err,~,~,~,~,N]=ap3(Uk,0,1);
% N=reshape(N,size(Uk));