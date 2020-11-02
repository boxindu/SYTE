function[U, y, time, Hc, Ur] = SYTE_P1_V2(Ac, K, B, alpha, b_uni, l)
%% Variant II of SyTE-Fast-P
%   --INPUTS:
%       -- Ac: a cell array of the adjacency matrices of input graphs
%       -- K: the number of input graphs
%       -- B: the anchor link tensor (a tensor)
%       -- alpha: the weighting parameter used in the formulation
%       -- b_uni: 0/1. 0: b is not uniform tensor; 1: b is uniform tensor (anchor links are unavailable)
%       -- l: Krylov subspace size
%   --OUTPUTS: 
%       -- U, y, Hc, Ur: the implicit solution of Variant II for SYTE-Fast-P
%       -- time: running time

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
    bb = -(1 - alpha).*B;
    H = []; I = [];
    U = {}; U1 = {}; Hc = {};
    for k = 1:K
        dk = sum(Ac{k,1},2).^(-0.5);
        dk(dk == Inf) = 0;
        Ak = bsxfun(@times, dk, Ac{k,1});
        Ak = bsxfun(@times, Ak, dk');
%         bk = ones(g_size_list(k), 1);
        if k == 1
            bk = (-(1 - alpha)).* ones(g_size_list(k), 1);
%             bk = ones(g_size_list(k), 1);
        else
            bk = ones(g_size_list(k), 1);
        end
        [Uk1, Hk] = arnoldi(Ak,bk,l);
        Uk = Uk1(:,1:l); 
        if k == 1
            Ik = sparse(1:l, 1:l, 1);
            H = sparse(Hk(1:l,:)); I = Ik; U{1,1} = Uk; U1{1,1} = Uk' * ones(size(Uk',2),1);
            Hc{1,1} = H;
        else
            H = kron(H, sparse(Hk(1:l,:))); I = kron(I, Ik); U{k,1} = Uk; U1{k,1} = Uk' * ones(size(Uk',2),1);
            Hc{k, 1} = sparse(Hk(1:l, :));
        end
    end
    fprintf('size of H, I: %d, %d', size(H,1), size(I,1));
    for i = 1:K
        if i == 1
            Ur = U1{1,1};
        else
            Ur = kron(Ur, U1{i,1});
        end
    end
    lin = tic;
%     opts.UHESS = true;
%     y = linsolve(I - (alpha.* H), (-(1 - alpha)).* Ur, opts);
%     y = mldivide(I - (alpha.* H), (-(1 - alpha)).* Ur);
    [P, z, ~] = Acc2_4_acc1(Hc, K, Ur, alpha);
    [y] = calculate_x(P, z, l, 1, l, l, l);
    fprintf('linsolve time: %d\n', toc(lin));

else
    % When b is not uniform, b is a low-rank tensor
    bnz = find(B == 1);
    bb = {}; 
    U = {}; y = {}; bk = {};
    for k = 1:K
        dk = sum(Ac{k,1},2).^(-0.5);
        dk(dk == Inf) = 0;
        Ak{k,1} = bsxfun(@times, dk, Ac{k,1});
        Ak{k,1} = bsxfun(@times, Ak{k,1}, dk');
    end
    for i = 1: size(bnz, 1)
        bb{1,i} = sparse(bnz(i,3), 1, 1, g_size_list(1), 1);
        bb{2,i} = sparse(bnz(i,2), 1, 1, g_size_list(2), 1);
        bb{3,i} = sparse(bnz(i,1), 1, 1, g_size_list(3), 1);
        
        H = []; I = []; U2 = {}; U1 = {}; Hc = {};
        for k = 1:K
%             dk = sum(Ac{k,1},2).^(-0.5);
%             Ak = bsxfun(@times, dk, Ac{k,1});
%             Ak = bsxfun(@times, Ak, dk');
%             Ak = alpha^(1/K).* Ak;
            if k == 1
                bk = (-(1 - alpha)).* bb{1,i};
            else
                bk = bb{k, i};
            end
            [Uk1, Hk] = arnoldi(Ak{k,1},bk,l);
            if ismember(Inf, Hk) || ismember(NaN, Hk)
                fprintf('bad condition');
            end
            Uk = Uk1(:,1:l);
            if k == 1
                Ik = sparse(1:l, 1:l, 1);
                H = sparse(Hk(1:l,:)); 
                Hc{1,1} = H;
%                 H = Hk(1:l, :);
                U1{1,1} = Uk' * bb{1,i}; I = Ik; U2{1,1} = Uk;
            else
                H = kron(H, sparse(Hk(1:l, :))); 
                Hc{k,1} = sparse(Hk(1:l, :));
%                 H = kron(H, Hk(1:l, :));
                U1{k,1} = Uk' * bb{k,i}; I = kron(I, Ik); U2{k,1} = Uk;
            end
        end
        for j = 1:K
            if j == 1
                Ur = U1{1,1};
            else
                Ur = kron(Ur, U1{j, 1});
            end
        end
        lin = tic;
%         opts.RECT = true;
%         y2 = linsolve(I - H, (-(1 - alpha)).* Ur, opts);
        [P, z, ~] = Acc2_4_acc1(Hc, K, Ur, alpha);
        [y2] = calculate_x(P, z, l, 1, l, l, l);
%         y2 = mldivide(I - H, (-(1 - alpha)).* Ur);
        fprintf('linsolve time: %d\n', toc(lin));
        U{i,1} = U2;
        y{i,1} = y2;
    end
end

%% for three input graphs only:
% ax1 = size(Ac{1,1},1);
% ax2 = size(Ac{2,1},1);
% ax3 = size(Ac{3,1},1);
% [x] = calculate_x(U, y, 4, b_uni, ax1, ax2, ax3); 
% X = reshape(x, filp(g_size_list));
% X = (-1) * X;

time = toc(acc1_time);
fprintf('Acc1_v2 running time: %d\n', toc(acc1_time));
end
