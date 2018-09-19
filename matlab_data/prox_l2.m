% Solve soft thresholding with l2 norm
function X=prox_l2(T,mu)
    X = zeros(size(T));
    for i=1:size(T,1)
        t=T(i,:);
        nrm=norm(t,2);
        if nrm<mu
            x=zeros(size(t));
        else
            x=(1-mu/nrm)*t;
        end
        X(i,:)=x;
    end
end