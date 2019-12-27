function [y0, alpha,results, findY0_time,qp_time, findY_time]= RMVC(candidate_Yi,baseline_Y0,nCluster,gnd)
%
%  ========================================================================
% candidate_Yi: an array, contains normalized cluster indicator matrix from
%               multi-view learners or results of single learners from all
%               views
% baseline_Y0: an array, normalized cluster indicator matrix from each view
%              but computed by single-view algorithms


Yi = candidate_Yi;
Y0 = baseline_Y0;
nLearner = size(Yi,3);
viewNum = size(Y0,3);
nSmp = size(Y0,1);

fprintf('preparing...\n');
%%%%  vectorize clustering indicator matrix into vectors;
f0 = zeros(nSmp*nSmp,viewNum);
for v = 1:viewNum
    f0(:,v) = reshape(Y0(:,:,v)*Y0(:,:,v)',nSmp*nSmp,1);
end
fi = zeros(nSmp*nSmp,nLearner);
for l = 1:nLearner
    fi(:,l) = reshape(Yi(:,:,l)*Yi(:,:,l)',nSmp*nSmp,1);
end


%% to find the best single-view clustering result
tic;
Dis = zeros(viewNum,nLearner);
for v = 1:viewNum
    for l = 1:nLearner
        temp = -2*f0(:,v)'*fi(:,l);
        Dis(v,l) = f0(:,v)'*f0(:,v) + temp;
    end
end
[MinDistVal,MinIdx] = min(Dis);
if length(unique(MinIdx)) == 1
    fprintf('find y0 in original views.\n')
    y0 = f0(:,unique(MinIdx));
    resnorm = 0;
else
    fprintf('to find a equivalent y0...\n')
    [y0, resnorm,P, q] = solveY0_byLR(fi, MinDistVal);
end
findY0_time = toc;
clear P q resnorm

fprintf('solving the QP problem ...\n');
tic;
[fstar, alpha]= QPsolver(fi,y0);
qp_time = toc;

fprintf('finding final cluster indicator...\n');
A = reshape(fstar,nSmp,nSmp);
A = max(A,A');
%%%% A = Y_star*Y_star', now solve the following problem to get Y_star
%%%% min ||Y*Y'-A||_F^2, s.t. Y'*Y = I, Y >= 0
%%%% <=> max Tr(Y'*A*Y), s.t. Y'*Y = I, Y >= 0
distArray = zeros(nLearner,1);
for i = 1:nLearner
    distArray(i) = MinDistVal(i) + fi(:,i)'*fi(:,i);
end
[results, findY_time] = findReliableY(A,nCluster,gnd,distArray,Yi,alpha);

end






