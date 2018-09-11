function Z =makeblk(H,Uk,Zo,index)
    col_full = find(sum(H~=0));
    U = Uk*Zo;
    k = size(Uk,2);
    trial = 0;
    if length(col_full)== k
        Z=Zo;
    else
        while (trial < min(k,5))
            row_id = find(H(:,col_full(index))>0);
            row_ic = find(H(:,col_full(index))==0);
            if length(row_id)==1
                index=datasample(1:length(col_full),1);
                trial=trial+1;
                continue;
            end
            [s,D,v] = svd(U(row_id,:));
            d = diag(D);
            rk = 1;
            while (d(rk+1)/d(rk)>2e-2)
                rk = rk+1;
                if rk+1>size(d,1)
                    break;
                end
            end
            if rk==1
                Z=Zo;
                break;
            else
                if (rk == k) 
                    index=datasample(1:length(col_full),1);
                    trial = trial+1;
                    continue;
                end
                
%                 col_else = col_full;
%                 col_else(index)=[];
%                 if max(sum(abs(U(row_id,col_else))))>1e-3*length(row_id)
%                     index=datasample(1:length(col_full),1);
%                     trial=trial+1;
%                     continue;
%                 end
                if (abs(d(rk)-1)>1e-3)
                    index=datasample(1:length(col_full),1);
                    trial=trial+1;
                    continue;
                end


                %[~,column_id]=sort(sum(abs(v(:,1:rk)),2),'descend');
                column_id = find(sum(abs(v(:,1:rk)),2)>1e-4);
                %col_keep = column_id(column_id~=col_full(index));
                col_keep = column_id(ismember(column_id,find(sum(H)==0)));
                %col_keep = col_keep(1:rk-1);
                col_keep = datasample(col_keep,rk-1,'Replace',false);

                %vv=sparse([col_keep;col_full(index)],1:rk,ones(1,rk),k,rk,rk);
                vv = zeros(k,rk);
                vv([col_keep;col_full(index)],:) = v([col_keep;col_full(index)],1:rk);
                U(row_id,:)=s(:,1:rk)*D(1:rk,1:rk)*vv';
                U(row_ic,col_keep)=0;
                
                [S,~,V]=svd(Uk'*U,0);
                Z = S*V';
                break;
            end
        end
    end
    
end