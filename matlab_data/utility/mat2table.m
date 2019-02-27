function out = mat2table(A,Rh,Ch,precision,method)
% Make a table in latex format.
% Input:
%  A: Data, m x n matrix. 
% Rh: Row header, 1 x n+1 cell
% Ch: Column header, m x 1 cell
% minmax: Bf the font of maximum or minimum column value

% Output: 
% out: a table, m+1 x n+1

% Feiyu Chen, 2016
% Last modified, Yuchen Yang, 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Create a table
% Rh = {'Algorithms','K-means1*','K-means','Kind','Katt + Kmns'};
% Ch = {'USPS';'MNIST';'COIL20';'COIL100';...;
%     'Yale';'YaleB';'ORL';'PIE';'Reuters';'TDT2'};
% precision = 3;
% 
% outT = mat2table(T(:,1:4),Rh,Ch,precision);
% disp(outT)
% outF = mat2table(F(:,1:4),Rh,Ch,precision);
% disp(outF)
% outAC = mat2table(AC(:,1:4),Rh,Ch,precision);
% disp(outAC)
% outNMI = mat2table(NMI(:,1:4),Rh,Ch,precision);
% disp(outNMI)
%！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
% For example:
% Rh = {'Table name','Row1','Row2','Row3'};
% Ch = {'Column1';'Column2';'Column3'};  
% T = randn(size(Ch,1),size(Rh,2)-1);
% 
% outT = mat2table(T,Rh,Ch);
% disp(outT)

% Result:
% \begin{tabular}{|c|c|c|c|}
% \hline
% Table name & Row1 & Row2 & Row3  \\ \hline \hline
% Column1 &   -0.102 & 0.313 & -0.165 \\ \hline
% Column2 &   -0.241 & -0.865 & 0.628 \\ \hline
% Column3 &   0.319 & -0.030 & 1.093 \\ \hline
% \end{tabular}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3
    error('Not enough input,need data,row name and column name')
end
if nargin == 3
    precision = '4';
    bfmethod = '';
else
    if nargin == 4
        bfmethod = '';
    else
        bfmethod = method;
    end
    precision = int2str(precision);
end


% Define the format of output.
out_num = [' %0.' precision 'f &'];
% Rounding criterion.
z = zeros(1, str2double(precision) + 1);
z(1) = '.'; z(2 : end) = '0'; z = char(z);

[m,n] = size(A);

% determin the size of '{|c|c|c|}'.
nc = sprintf('|');
for i = 1:n+1
nc = [nc sprintf('c|')];
end

% Row header
R = [];
for i = 1:n+1 
R = [R sprintf('%s & ',Rh{i})];
end
% delete the last '& '.
R = R(1 : end - 2);

% first row
out = sprintf('\\begin{tabular}{%s}\n\\hline\n%s \\\\ \\hline \\hline', nc,R);
%% table
for i = 1 : m
    out = [out sprintf('\n%s &  ', Ch{i})];  %#ok<*AGROW>
    bf_id = 0;
    if ~strcmp(bfmethod,'')
        if strcmp(bfmethod,'max')
            bfizer = max(A(i,:));
        end
        if strcmp(bfmethod,'min')
            bfizer = min(A(i,:));
        end
        bf_id = find(abs((A(i,:)- bfizer))<10^(-precision-2));
    end
    for j = 1 : n
        if ismember(j,bf_id)      
            temp = [sprintf('{\\bf  ') sprintf(['%0.' precision 'f'],A(i, j)), sprintf('} &')];
        else
            temp = sprintf(out_num, A(i, j));
        end
        % Rounding:1.0001 = 1
%         dot_position = find(temp == '.');
%         if temp(dot_position : end - 2) == z
%             temp = temp(1 : dot_position - 1);
%             temp = [temp ' &'];   
%         end
        out = [out temp];
    end
    % delete the last '&'.
    out = out(1 : end - 1);
    % add  '\\ \hline' at the end of each row.
    out = [out '\\ \hline'];
end
% last row
out = [out sprintf('\n\\end{tabular}')];