function [Y, weight]= QPsolver(candidate_prediction,baseline_prediction)

candidate_num = size(candidate_prediction,2);
H = candidate_prediction'*candidate_prediction*2;
f = -2 * candidate_prediction' * baseline_prediction;

Aeq = ones(1,candidate_num);
beq = 1;
lb = zeros(candidate_num,1);
ub = ones(candidate_num,1);

options=optimset('Algorithm', 'interior-point-convex','Display','off');

% use quadprogramming to get the final result.
% it's faster to use mosek
[weight,fval] = quadprog(H,f,[],[],Aeq, beq, lb, ub,[],options);

Y = 0;
for i = 1:candidate_num
    Y = Y + weight(i)*candidate_prediction(:,i);
end
end