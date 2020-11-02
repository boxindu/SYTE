function[Q, y, time] = Acc2_4_acc1(Ac, K, b, alpha)
%% The altered second accelerated algorithm (StTE-Fast-P*) used for Variant II of SyTE-Fast-P
%   --INPUTS: 
%       -- Ac: a cell array of the adjacency matrices of input graphs
%       -- K: the number of input graphs
%       -- b: the vectorized anchor link tensor (a vector)
%       -- alpha: the weighting parameter used in the formulation
%   --OUTPUTS: 
%       -- U, y: the implicit solution of SYTE-Fast-P
%       -- time: running time

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
Lamb = []; Q = {}; ran = zeros(K,1);
for k = 1:K
%     d = sum(Ac{k,1},2).^(-0.5);
%     Ak = bsxfun(@times, d, Ac{k,1});
%     Ak = bsxfun(@times, Ak, d');
    Ak = Ac{k, 1};
    eig = tic();
    ran(k,1) = rank(full(Ak));
    [Qk, Lambk] = eigs(alpha^(1/K) .* Ak, ran(k,1));
%     [Qk, Lambk] = eigs(Ak, r);
    fprintf('eigs time: %d\n', toc(eig));
    if k == 1
        Lamb = sparse(Lambk);
        Q{1, 1} = Qk;
    else
        Lamb = kron(Lamb, sparse(Lambk));
        Q{k, 1} = Qk;    % NEEDS MODIFICATION !
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
c = reshape(c, prod(ran), 1);
% c = Q' * b;
Ay = sparse(1:prod(ran), 1:prod(ran), 1) - Lamb;
small_sys = tic;
y = mldivide(Ay, c);
fprintf('time for small sys: %d\n', toc(small_sys));
% x = Q * y;
time = toc(acc2_time);
fprintf('Acc2 running time: %d\n', time);
% X = reshape(x, filp(g_size_list));

end