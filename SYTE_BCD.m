function[xc, b_offdiag, solu, time] = SYTE_BCD(Ac, Nc, K, B, alpha, b_uni, l, max_iter, explicit_output)
%% SyTE-BCD algorithm which uses complete block coordinate descent (for attributed graphs)
%   --INPUTS:
%       -- Ac: a cell array of the adjacency matrices of input graphs. Ac{1,1}
%       = A1, Ac{2,1} = A2, etc. 
%       -- Nc: a cell array of node attributes. Nc{i,1} is n_i-by-Q where n_i
%       is the number of nodes in the i-th graph, and Q is the number of
%       categorical node attributes. Each row has 0/1 elements with 1
%       indicating the existence of the corresponding attribute;
%       -- K: the number of input graphs
%       -- B: the anchor link tensor (a tensor)
%       -- alpha: the weighting parameter used in the formulation
%       -- b_uni: 0/1. 0: b is not uniform tensor; 1: b is uniform tensor (anchor links are unavailable)
%       -- l: the Krylov subspace size
%       -- max_iter: maximum iteration number
%       -- explicit_output: 0/1. 0: output implicit output; 1: output tensor
%       output (this option might be time and space consuming!)
%   --OUTPUTS:
%       -- xc, b_offdiag: the implicit solution of SYTE-BCD
%       -- time: running time
%       -- solu: the explicit solution if 'explicit_output' is 1. 
%       It is the vectorized tensor with the high-order scores (similarity tensor).
%       Note that (3D for example) element X(i,j,k) is the similarity score of 
%       node i in G3, node j in G2, and node k in A; if G1 has n1 nodes; G2 has 
%       n2 nodes, G3 has n3 nodes, then X is an n3-by-n2-by-n1 tensor.

acc1_attr = tic;
if isempty(Nc)
    fprintf('Node attr cell array should not be empty! Check function inputs.');
end
sys_dim = 1;
num_na = size(Nc{1,1},2);
Ac_na = {}; B_na = {}; N_na = {};
if b_uni == 0 % when B is not uniform
    parfor i = 1 : K
        for j = 1 : num_na
            for k = 1 : num_na
                node_in_j = find(Nc{i, 1}(:, j) == 1);
%                 node_in_k = find(Nc{i, 1}(:, k) == 1);
                Nj = sparse(node_in_j, node_in_j, 1, size(Ac{i, 1},1), size(Ac{i, 1},1));
%                 Nk = sparse(node_in_k, node_in_k, 1, size(Ac{i, 1},1), size(Ac{i, 1},1));
                Ac_na{i,j,k} = bsxfun(@times, bsxfun(@times,Nc{i,1}(:,j), Ac{i,1}), Nc{i,1}(:,k)');
                
                N_na{i, j} = Nj; 
%                 Ac_na{i, j, k} = Nj * Ac{i, 1} * Nk;
            end
        end
        sys_dim = sys_dim * size(Ac{i, 1}, 1);
    end
    
    for i = 1 : num_na
        for j = 1 : K
            if j == 1
                B_na{i, 1} = ttm(B, N_na{K - j + 1, i}, j);
            else
                B_na{i, 1} = ttm(B_na{i, 1}, N_na{K - j + 1, i}, j);
            end
        end
    end
    for i = 1 : num_na
        if i == 1
            b_diag = reshape(B_na{i, 1}, [sys_dim, 1]);
            b_diag = double(b_diag);
        else
            b_diag = b_diag + double(reshape(B_na{i, 1}, [sys_dim, 1]));
        end
    end
    b_offdiag = sparse(double(reshape(B, [sys_dim, 1])) - b_diag);
    % Begin solving num_na equations
    fprintf('Begin solving num_na equations...');
    ac_temp = {}; xc = {}; iter = 0;
    x_temp = sparse(sys_dim, 1);
    
    for i = 1 : num_na
        for j = 1:K
            ac_temp{i,j} = Ac_na{j,i,i};
        end
    end
    while iter < max_iter
        fprintf('iter %d \n', iter);
        iter = iter + 1;
        if iter == 1
            for i = 1 : num_na
                ac{1,1} = ac_temp{i,1}; ac{2,1}=ac_temp{i,2}; ac{3,1}=ac_temp{i,3};
                %% Use Variant II of SYTE-Fast-P:
                [U, y, ~] = SYTE_P1_V2(ac, K, B_na{i, 1}, alpha, b_uni, l);
%                 cal_x = tic;
                [x] = calculate_x(U, y, l, 0, size(Ac{1,1},1), size(Ac{2,1},1), size(Ac{3,1},1)); 
                x_temp = x_temp + (-1).*x;
                xc{i,1} = (-1).*x;   
                %% Use FP
%                 b_temp = reshape(B_na{i,1}, [size(Ac{1,1},1)*size(Ac{2,1},1)*size(Ac{3,1},1),1]);
%                 b_temp = double(b_temp); b_temp = sparse(b_temp);
%                 clear B;
%                 [x, ~] = FP(ac, K, Nc, b_temp, alpha, 20, 0.001);
%                 xc{i,1} = x;
% %                 x_temp = x_temp + x;

            end
        else
%             x_sum = sptensor([size(Ac{3,1},1), size(Ac{2,1},1),size(Ac{3,1},1)]);
            for i = 1 : num_na
%                 x_sum = sptensor([size(Ac{3,1},1), size(Ac{2,1},1),size(Ac{1,1},1)]);
                x_sum = sptensor([size(Ac{3,1},1), size(Ac{2,1},1),size(Ac{1,1},1)]);
                for j = 1 : num_na
                    if j ~= i
                        x_tenor = sptensor(reshape(full(xc{j,1}), size(Ac{3,1},1), size(Ac{2,1},1),size(Ac{1,1},1)));
                        for k = 1 : K
                            % ND sparse array should be used below!
                            d1_temp = sum(Ac_na{K-k+1, i, j},2).^(-0.5);
                            d2_temp = sum(Ac_na{K-k+1, i, j},1).^(-0.5);
                            d1_temp(d1_temp==Inf) = 0; d2_temp(d2_temp==Inf) = 0;
                            Ac_temp = bsxfun(@times, d1_temp, Ac_na{K-k+1, i, j});
                            Ac_temp = bsxfun(@times, Ac_temp, d2_temp);
                            x_tenor = ttm(x_tenor, Ac_temp, k); % A should be normalized!!!
                         end
                        x_sum = x_sum + x_tenor;
                    end
                end
                B_temp = B_na{i, 1} + alpha .* x_sum; 
                ac{1,1} = ac_temp{i,1}; ac{2,1}=ac_temp{i,2}; ac{3,1}=ac_temp{i,3};
                %% Use Variant II of SYTE_P1_V2
%                 [U, y, ~] = SYTE_P1_V2(ac, K, B_temp, alpha, b_uni, l);
%                     cal_x1 = tic;
%                 [x] = calculate_x(U, y, l, 0, size(Ac{1,1},1), size(Ac{2,1},1), size(Ac{3,1},1)); 
%                 x_temp = x_temp + (-1).*x;
%                 xc{i,1} = (-1).*x;
                %% Use FP
                b_temp = reshape(B_temp, [size(Ac{1,1},1)*size(Ac{2,1},1)*size(Ac{3,1},1),1]);
                b_temp = double(b_temp); b_temp = sparse(b_temp);
                clear B_temp;
                [x, ~] = FP(ac, K, [], b_temp, alpha, 20, 0.001);
%                 x_temp = x_temp + x;
                xc{i,1} = x;

            end
        end
    end
if explicit_output
    for i = 1 : num_na
        x_temp = x_temp + xc{i,1};
    end
    solu = x_temp + b_offdiag;
else
    solu = 0; % dummy output
end

% Uniform B is not considered in this function 
end

time = toc(acc1_attr);
% x is calcualted explicitly only for evaluation.
% fprintf('runrime for acc1_attr: %d \n', time - time_cal_x * num_na); 
end