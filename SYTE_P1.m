function[U, y, time, X] = SYTE_P1(Ac, K, b, alpha, b_uni, l, explicit_output)
%% Accelerated algorithm SyTE-Fast-P for Tensor Sylvester equation on plain graphs
%   --INPUTS:
%       -- Ac: a cell array of the adjacency matrices of input graphs
%       -- K: the number of input graphs
%       -- b: the vectorized anchor link tensor (a vector)
%       -- alpha: the weighting parameter used in the formulation
%       -- b_uni: 0/1. 0: b is not uniform vector; 1: b is uniform vector (anchor links are unavailable)
%       -- l: the Krylov subspace size
%       -- explicit_output: 0/1. 0: output implicit output; 1: output tensor
%       output (this option might be time and space consuming!)
%   --OUTPUTS:
%       -- U, y: the implicit solution of SYTE-Fast-P
%       -- time: running time
%       -- X: the explicit solution if 'explicit_output' is 1

acc1_time = tic;
if size(Ac,1) < 2
    printf('The # of input graphs is less than 2!');
end
sys_dim = 1;
for i = 1:K
    sys_dim = sys_dim * size(Ac{i,1},1);
    if i == 1
        g_size_list = [size(Ac{i,1},1)];
    else
        g_size_list = [g_size_list, size(Ac{i,1},1)];
    end
end
if b_uni == 1
    % Now b is a VECTOR !
    b = -(1 - alpha).*b;
    H = []; I = []; 
    U = {}; U1 = {};
    for k = 1:K
        dk = sum(Ac{k,1},2).^(-0.5);
        Ak = bsxfun(@times, dk, Ac{k,1});
        Ak = bsxfun(@times, Ak, dk');
        Ak = alpha^(1/K).* Ak;
%         bk = ones(g_size_list(k), 1);
        if k == 1
            bk = (-(1 - alpha)).* ones(g_size_list(k), 1);
        else
            bk = ones(g_size_list(k), 1);
        end
        [Uk1, Hk] = arnoldi(Ak,bk,l);
        Uk = Uk1(:,1:l);
        if k == 1
            Ik = cat(1, sparse(1:l, 1:l, 1), zeros(1,l));
            H = Hk; U1{1,1} = Uk1' * ones(size(Uk1',2),1); I = Ik; U{1,1} = Uk;
        else
            H = kron(H, Hk); U1{k,1} = Uk1' * ones(size(Uk1',2),1); I = kron(I, Ik); U{k,1} = Uk;
        end
    end

    opts.RECT = true;
    for i = 1:K
        if i == 1
            Ur = U1{1,1};
        else
            Ur = kron(Ur, U1{i,1});
        end
    end
    
    lin = tic;
    y = linsolve(I - H, Ur, opts);
    fprintf('linsolve time: %d\n', toc(lin));
    if explicit_output 
        r0 = b;
        x0 = zeros(sys_dim, 1);
        r0 = r0 - U * y + U1 * H * y;
        nr = norm(r0);
        x0 = x0 + U * y;
        fprintf('residual norm: %d\n', nr);
    end
else
    % When b is not uniform, b is a low-rank tensor !
    bnz = find(b == 1);
    bb = {}; 
    x0 = [];
    U = {}; y = {};
    for i = 1: size(bnz, 1)
        bb{1,i} = sparse(bnz(i,1), 1, 1, g_size_list(1), 1);
        bb{2,i} = sparse(bnz(i,2), 1, 1, g_size_list(2), 1);
        bb{3,i} = sparse(bnz(i,3), 1, 1, g_size_list(3), 1);
%         b_temp = kron(b3{i}, b2{i});
%         b_temp = kron(b_temp, b1{i});
%         b_temp = -(1 - alpha).*b_temp;
        
        H = []; I = []; U2 = {}; U1 = {};
        for k = 1:K
            dk = sum(Ac{k,1},2).^(-0.5);
            Ak = bsxfun(@times, dk, Ac{k,1});
            Ak = bsxfun(@times, Ak, dk');
            Ak = alpha^(1/K).* Ak;
%             if k == 1
%                 bk = (-(1 - alpha)).* bb{K,i};
%             else
%                 bk = bb{K-k+1, i};
%             end
            if k == 1
                bk = (-(1 - alpha)).* bb{1,i};
            else
                bk = bb{k, i};
            end
            [Uk1, Hk] = arnoldi(Ak,bk,l);
            Uk = Uk1(:,1:l);
            if k == 1
                Ik = cat(1, sparse(1:l, 1:l, 1), zeros(1,l));
                H = Hk; U1{1,1} = Uk1' * bb{1,i}; I = Ik; U2{1,1} = Uk;
            else
                H = kron(H, Hk); U1{k,1} = Uk1' * bb{k,i}; I = kron(I, Ik); U2{k,1} = Uk;
            end
        end
        x1 = zeros(sys_dim, 1);  
        opts.RECT = true;
        for j = 1:K
            if j == 1
                Ur = U1{1,1};
            else
                Ur = kron(Ur, U1{j, 1});
            end
        end
        lin = tic;
        y2 = linsolve(I - H, Ur, opts);
        fprintf('linsolve time: %d\n', toc(lin));
        U{i,1} = U2;
        y{i,1} = y2;
        
        if explicit_output
            x1 = x1 + U * y;
            x0 = x0 + x1;
        end
    end
end
if explicit_output
    x = x0;
    X = reshape(x, filp(g_size_list));
else
    X = 0;
end
time = toc(acc1_time);
fprintf('Acc1 running time: %d\n', toc(acc1_time));
    
end
