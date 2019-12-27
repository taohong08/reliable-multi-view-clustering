function [results, findY_time,objArray]  = findReliableY(A,cluster_num,gnd,distArray,Yi,alpha)
n = size(A,1);
[v, d] = eig(A);
d = diag(d);
[temp, idx] = sort(d, 'descend');
G0 = v(:,idx(1:cluster_num));

reptime = 1;
results = zeros(reptime,8);
findY_time =zeros(reptime,1);
objArray = zeros(reptime,1);
redDist = zeros(reptime,1);
for i = 1:reptime
    tic;
    D = eye(n);
    [Gr] = SpectralRotation(G0);
    [G] = NMFdiscrete(Gr+0.2, D, A);
    [~,y] = max(G,[],2);
    clear Gr G
    
    findY_time(i,1) = toc;
    Y = zeros(n,length(unique(y)));
    for j = 1:length(unique(y))
        Y(find(y==j),j) = 1;
    end
    Y = Y./repmat(sqrt(sum(Y)),size(Y,1),1);
    
    fi = zeros(n*n,size(Yi,3));
    for l = 1:size(Yi,3)
        fi(:,l) = reshape(Yi(:,:,l)*Yi(:,:,l)',n*n,1);
    end
    objArray(i) = calObj(distArray,fi,Y,alpha);
    clear fi;
    results(i,:) = ClusteringMeasure(gnd,y);
end

end

function [obj] = calObj(q,fi,Y,alpha)
% Y = Y./repmat(sqrt(sum(Y)),size(Y,1),1);
f = reshape(Y*Y',size(Y,1)^2,1);
tempDist = EuDist2(f',fi');
obj = 0;
for i = 1:length(alpha)
    obj = obj + alpha(i)*(q(i) - tempDist(i));
end
end


% Nonnegative relaxation for clustering
% max trace(Q'*A*Q)|Q'*D*Q=I or max trace(Q'*An*Q)|Q'*Q=I
function [Qd obj orobj objhard] = NMFdiscrete(Q, D, A)

ITER = 200;
[total_num, class_num] = size(Q);
obj = zeros(ITER,1);
orobj = zeros(ITER,1);
objhard = zeros(ITER,1);
for iter = 1:ITER
    
    Q = Q*diag(sqrt(1./diag(Q'*D*Q)));
    QQ = Q'*D*Q;
    Lamda = Q'*A*Q;
    Lamda = (Lamda + Lamda')/2;
    
    QQI = QQ - eye(class_num);
    obj(iter) = trace(Lamda) - trace(Lamda*(QQI));
    orobj(iter) = sqrt(trace(QQI'*QQI)/(class_num*(class_num-1)));
    
    [dumb res] = max(Q,[],2);
    Fr = zeros(total_num, class_num);
    for cn = 1:class_num
        Fr((res==cn),cn) = 1;
    end;
    Fr = Fr*diag(sqrt(1./diag(Fr'*Fr)));
    objhard(iter) = trace(Fr'*A*Fr);
    
    QA = Q'*A*Q;
    S = (A*Q + eps)./(D*Q*QA + eps);
    S = S.^(1/2);
    Q = Q.*S;
    Q = Q*diag(sqrt(1./diag(Q'*D*Q)));
    
end;

Qd = Q;
end

function [Fr obj Q] = SpectralRotation(F)

[n,c] = size(F);

F(sum(abs(F),2) <= 10^-18,:) = 1;
F = diag(diag(F*F').^(-1/2)) * F;

con_flag = 0;
Q = orth(rand(c));
obj_old = 10^10;
for iter = 1:30
    M = F*Q;
    G = binarizeM(M, 'max');
    
    aa = M - G; obj = trace(aa'*aa);
    if (obj_old - obj)/obj < 0.000001
        con_flag = 1;
        break;
    end;
    obj_old = obj;
    
    [U, d, V] = svd(F'*G);
    Q = U*V';
end;
Fr = G;

if con_flag == 0
    warning('does not converge');
end;
end




function B = binarizeM(M, type)
% binarize matrix M to 0 or 1

[n,c] = size(M);

B = zeros(n,c);

if strcmp(type, 'median')
    B(find(M > 0.5)) = 1;
else
    
    if strcmp(type, 'min')
        [temp idx] = min(M,[],2);
    elseif strcmp(type, 'max')
        [temp idx] = max(M,[],2);
    end;
    
    for i = 1:n
        B(i,idx(i)) = 1;
    end;
    
end;
end
