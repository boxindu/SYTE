function[Q, y, time, X] = SYTE_P2(Ac, K, r, b, alpha, explicit_output)
%% The second accelerated algorithm (SyTE-Fast-P*) for Tensor Sylvester equation on plain graphs
%   --INPUTS: 
%       -- Ac: a cell array of the adjacency matrices of input graphs
%       -- K: the number of input graphs
%       -- b: the vectorized anchor link tensor (a vector)
%       -- alpha: the weighting parameter used in the formulation
%       -- r: the approximation rank of eigen-decomposition 
%       -- explicit_output: 0/1. 0: output implicit output; 1: output tensor
%       output (this option might be time and space consuming!)
%   --OUTPUTS: 
%       -- U, y: the implicit solution of SYTE-Fast-P
%       -- time: running time
%       -- X: the explicit solution if 'explicit_output' is 1

acc2_time = tic;
if size(Ac,1) < 2
    printf('The # of input graphs is less than 2!');
end
for i = 1:K
    if i == 1
        g_size_list = [size(Ac{i,1},1)];
    else
        g_size_list = [g_size_list, size(Ac{i,1},1)];
    end
end
Lamb = []; Q = {};
for k = 1:K
    d = sum(Ac{k,1},2).^(-0.5);
    d(d == Inf) = 0;
    Ak = bsxfun(@times, d, Ac{k,1});
    Ak = bsxfun(@times, Ak, d');
    eigtime = tic();
    [Qk, Lambk] = eigs(alpha^(1/K) .* Ak, r);
%     [Qk, Lambk] = eigs(Ak, r);
    fprintf('eigs time: %d\n', toc(eigtime));
    if k == 1
        Lamb = sparse(Lambk);
        Q{1, 1} = Qk;
    else
        Lamb = kron(Lamb, sparse(Lambk));
        Q{k, 1} = Qk;    
    end
end
b = -(1 - alpha).*b;
for k = 1:K-1
    if k == 1
        Q_temp_dim = g_size_list(1);
        Q_temp = Q{1,1};
    else
        Q_temp_dim = Q_temp_dim * g_size_list(k);
        Q_temp = kron(Q_temp, Q{k, 1});
    end
end
c = Q{K,1}' * reshape(b, g_size_list(K), Q_temp_dim) * Q_temp;
c = reshape(c, r^K, 1);
% c = Q' * b;
Ay = sparse(1:r^K, 1:r^K, 1) - Lamb;
small_sys = tic;
y = mldivide(Ay, c);
fprintf('time for small sys: %d\n', toc(small_sys));

if explicit_output
%     x = kron(Q{1,1}, kron(Q{2,1}, Q{3,1})) * y;
    x = Q * y;
    X = reshape(x, filp(g_size_list));
else
    X = 0; % dummy output
end
time = toc(acc2_time);
fprintf('Acc2 running time: %d\n', time);

end