% function results = ClusteringMeasure(true_labels,cluster_labels,beta)
% results = [NMI,ACC, RandIndx,Purity, Fbeta, Precision, Recall, AdjRandIndx]
% This code calculates the mainly clustering mearsures (accuarcy, NMI, Rand Index, Purity, Fvalue) between two cluster partitions
% Date 29 March 2017(Hong Tao)

function results = ClusteringMeasure(true_labels,cluster_labels,beta)

if ~exist('beta','var')
    beta = 1;
end
true_labels = true_labels(:);
cluster_labels = cluster_labels(:);
[Purity cls] = purity(true_labels,cluster_labels);
[m n] = size(cls);
no_objects = sum(sum(cls));
i_dot = sum(cls,2);
j_dot = sum(cls,1);
tr = zeros(m,n);
for i = 1:m
     for j = 1:n
        if cls(i,j)<2
        tr (i,j) = 0;
        else
        tr (i,j) = nchoosek(cls(i,j),2);
        end
     end
end
term1 = sum(sum(tr));
for in = 1:length(i_dot)
    if i_dot(in)<2
       c_i(in) = 0;
    else
       c_i(in) = nchoosek(i_dot(in),2);  
    end
end

for jn = 1:length(j_dot)
    if j_dot(jn)<2
       c_j(jn) = 0;
    else
       c_j(jn) = nchoosek(j_dot(jn),2);  
    end
end

TP = sum(sum(tr));
FP = sum(c_i) - sum(sum(tr));
FN = sum(c_j) - sum(sum(tr));
TN = nchoosek(no_objects,2)-TP-FP-FN;
RandIndx =(TP+TN)/(TP+FP+FN+TN);
Precision = TP/(TP+FP);
Recall = TP/(TP+FN);
P = Precision;
R = Recall;
Fbeta = (beta^2 + 1)*P*R/(beta^2*P + R);

ARI_t = term1 - (sum(c_i)*sum(c_j))/nchoosek(no_objects,2);
ARI_d = 0.5*(sum(c_i) + sum(c_j)) - (sum(c_i)*sum(c_j))/nchoosek(no_objects,2);
AdjRandIndx = ARI_t/ARI_d;

ACC = accuracy(true_labels,cluster_labels)/100;
NMI = nmi(true_labels,cluster_labels);


results = [NMI,ACC, RandIndx,Purity, Fbeta, Precision, Recall,  AdjRandIndx ];
end

