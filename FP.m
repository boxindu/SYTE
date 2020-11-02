function[x, time] = FP(Ac, K, Nc, b, alpha, max_iter, tol)
%%  Fixed Point implementation for linear system of Sylvester Tensor equation
% Inputs: Ac: K-by-1 cell array, containing K adjacency matrix for K input
%            graphs. A_i is in A{i,1};
%         K: # of input graphs;
%         b: preference vector;
%         alpha: weighting parameter;
%         max_iter: # of maximum iteration;
%         tol: tolerance;
% Outputs: X: solution tensor;
%          time: running time;
if size(Ac,1) < 2
    printf('The # of input graphs is less than 2!')
end
sys_dim = 1;
for i = 1:K
    sys_dim = sys_dim * size(Ac{i,1},1);
end
A_Inc = [];
for i = 1:K-1
    if i == 1
        A_Inc = Ac{i,1};
    else
        A_Inc = kron(A_Inc, Ac{i,1});
    end
end
% calculate degree vector d & node attribute matrix N first, as inputs for 
if ~isempty(Nc)
    P = size(Nc{1,1},2); 
    d = spconvert([sys_dim, 1, 0]);
    tic;
    for p = 1: P 
        for k = 1:K
            if k == 1
                KronA = Ac{k,1} * Nc{k,1}(:, p);
            end
            KronA = kron(KronA, Ac{k,1} * Nc{k,1}(:,p));
        end
        d = d + KronA;
    end
    fprintf('Time for degree: %.2f sec\n', toc);     
    
end
N = spconvert([sys_dim, 1, 0]);
for p = 1: P
    for k = 1:K
        if k == 1
            KronN = Nc{k,1}(:,p);
        end
        KronN = kron(Kron, Nc{k,1}(:,p));
    end
    N = N + KronN;   % compute N as a kronecker similarity
end

end