function [ score G ] = purity(true_labels, cluster_labels )
%PURITY Computes clustering purity using the true and cluster labels and
%   return the value in 'score'.
%
%   Input  : true_labels    : N-by-1 vector containing true labels
%            cluster_labels : N-by-1 vector containing cluster labels

true_labels = true_labels(:);
cluster_labels = cluster_labels(:);
if size(true_labels) ~= size(cluster_labels)
    error('size(true_labels) must == size(cluster_labels)');
end
nSmp = length(true_labels);
L1 = unique(true_labels); 
nClass1 = length(L1);
L2 = unique(cluster_labels);
nClass2 = length(L2);
nClass = max(nClass1,nClass2);

%form contingency matrix
G = zeros(nClass);
for i = 1:nClass1
    for j = 1:nClass2
        G(i,j) = length(intersect(find(true_labels == L1(i)), find(cluster_labels == L2(j))));
    end
end
score = sum(max(G,[],2))/nSmp;



end

