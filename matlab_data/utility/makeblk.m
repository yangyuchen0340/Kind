function Z =makeblk(H,Uk,Zo,index)
%====================================================================
% Remove diagonality qualification of KindAP
% Select the required Z that generates block diagonals when projecting H to
% U
%
% Copyright: Yuchen Yang 2018
%====================================================================
% Input:
% Uk: n by k column-orthonormal matrix
%     (usually k eigenvectors of a Gram or Laplacian matrix)
% H: n by k matrix; column-orthogonal for nonzero columns
%     (usually rank deficient)
% Zo: the original projection from H to U by MATLAB default
% index: the column number the algorithm starts working on (usually random)
%====================================================================
% Output:
% Z: the desired rotation from H to U
%====================================================================
    % all nonzero columns of H
    col_full = find(sum(H~=0));
    % original U
    U = Uk*Zo;
    k = size(Uk,2);
    trial = 0;
    % return Zo for full-rank H
    if length(col_full)== k
        Z=Zo;
    else
        % check whether we can do operations on the certain column
        % a limit number of trials
        while (trial < min(k,5))
            % the rows of corresponding nonzero columns
            row_id = find(H(:,col_full(index))>0);
            row_ic = find(H(:,col_full(index))==0);
            % for a single element, retry 
            % can be improved later
            if length(row_id)==1
                index=datasample(1:length(col_full),1);
                trial=trial+1;
                continue;
            end
            % find the approximate rank of U(row_id,:), usually much smaller
            [s,D,v] = svd(U(row_id,:));
            d = diag(D);
            rk = 1;
            while (d(rk+1)/d(rk)>2e-2)
                rk = rk+1;
                if rk+1>size(d,1)
                    break;
                end
            end
            % rank==1 means a block diagonal has been found
            if rk==1
                Z=Zo;
                break;
            else
                if (rk == k) 
                    index=datasample(1:length(col_full),1);
                    trial = trial+1;
                    continue;
                end
                
                % check whether row_id are exactly the combination of
                % several clusters by looking at its singular values
                if (abs(d(rk)-1)>1e-3)
                    index=datasample(1:length(col_full),1);
                    trial=trial+1;
                    continue;
                end

                % if this column is doable, modify the loading matrix v to be sparse
                column_id = find(sum(abs(v(:,1:rk)),2)>1e-4);
                % keep rk-1 columns of all zero columns in H
                col_keep = column_id(ismember(column_id,find(sum(H)==0)));
                col_keep = datasample(col_keep,rk-1,'Replace',false);

                % vv=sparse([col_keep;col_full(index)],1:rk,ones(1,rk),k,rk,rk);
                vv = zeros(k,rk);
                % keep the original column
                vv([col_keep;col_full(index)],:) = v([col_keep;col_full(index)],1:rk);
                U(row_id,:)=s(:,1:rk)*D(1:rk,1:rk)*vv';
                U(row_ic,col_keep)=0;
                % project the modified U back to column Uk
                [S,~,V]=svd(Uk'*U,0);
                Z = S*V';
                break;
            end
        end
    end
    
end