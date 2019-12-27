function [y0, resnorm, P, q] = solveY0_byLR(fi, MinDistVal)
nSqrt = size(fi,1);
nLearner = size(fi,2);
P = [];
q = [];
fprintf('Forming the matrices of least squares\n');
for i = 1:nLearner
    for j = i+1:nLearner
        P = [P fi(:,i) - fi(:,j)];
        q = [q; (MinDistVal(j) - MinDistVal(i))];
    end
end
P = 2*P';
%% when the matrix size is not very large, we solve y0 by bounded LSR, otherwise by pinv
if nSqrt < 1e1 
    fprintf('begining the optimization with bounded least squares...\n');
    lb = zeros(nSqrt,1);
    ub = 0.5*ones(nSqrt,1);
    [y0, resnorm] = lsqlin(P,q,[],[],[],[],lb,ub); %% it's faster to use mosek
else
    fprintf('calculating A0 by pinv\n');
    y0 = P\q;
%     y0 = pinv(P)*q;
    resnorm = norm(P*y0-q);
end
fprintf('finishing finding A0\n');
end