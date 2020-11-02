function[x, map, time, U, y] = SYTE_A(Ac, Nc, K, B, alpha, explicit_output)
%% The SyTE-Fast-A algorithm for Sylvester Tensor Equation on attributed graphs
% REORDER ADJACENCY MATRIX; ONE ITERATION
% Use map to reorder gnd for evaluation
%   --INPUTS: 
%       -- Ac: a cell array of the adjacency matrices of input graphs
%       -- Nc: a cell array of node attributes.Nc{i,1} is n_i-by-Q where n_i
%       is the number of nodes in the i-th graph, and Q is the number of
%       categorical node attributes. Each row has 0/1 elements with 1
%       indicating the existence of the corresponding attribute;
%       -- K: the number of input graphs
%       -- B: the anchor link tensor (a tensor)
%       -- alpha: the weighting parameter used in the formulation
%       -- explicit_output: 0/1. 0: output implicit output; 1: output tensor
%       output (this option might be time and space consuming!)
%   --OUTPUTS: 
%       -- x: the explicit solution of SYTE-Fast-A. 
%       It is the vectorized tensor with the high-order scores (similarity tensor).
%       Note that (3D for example) element X(i,j,k) is the similarity score of 
%       node i in G3, node j in G2, and node k in A; if G1 has n1 nodes; G2 has 
%       n2 nodes, G3 has n3 nodes, then X is an n3-by-n2-by-n1 tensor.
%       -- time: running time
%       -- U, y: the implicit solution of SYTE-Fast-A.

acc1_attr = tic;
if isempty(Nc)
    fprintf('Node attr cell array should not be empty! Check function inputs.');
end
sys_dim = 1;
num_na = size(Nc{1,1},2);
Ac_na = {}; B_na = {}; N_na = {}; map = {}; x=[];
counter = zeros(K,1);
for j = 1 : num_na
    for i = 1 : K
        id = find(Nc{i,1}(:,j) == 1);
        if j == 1
            map{i} = [id, [1:size(id,1)]'];
            counter(i) = counter(i) + size(id,1);
            
        else
            map{i} = [map{i}; [id, [counter(i) + 1:counter(i) + size(id,1)]']];
            counter(i) = counter(i) + 1 + size(id,1);
        end
        permu = sparse(id, [1:size(id,1)]',1,size(Ac{i,1},1),size(Ac{i,1},1));
        ac{i,1} = permu * Ac{i,1} *permu';
        ac{i,1} = ac{i,1}(1:size(id,1), 1:size(id,1));
        if i == 1
            B_na = ttm(B, permu, K - i + 1);
            B_na = B_na(1:end,1:end,1:size(id,1));
        else
            B_na = ttm(B_na, permu, K - i + 1);
            if i == 2
                B_na = B_na(1:end,1:size(id,1),1:end);
            end
            if i == 3
                B_na = B_na(1:size(id,1),1:end,1:end);
            end
        end
    end
    %% Use FP
    if explicit_output
        b_temp = reshape(B_na, [size(Ac{1,1},1)*size(Ac{2,1},1)*size(Ac{3,1},1),1]);
        b_temp = double(b_temp); b_temp = sparse(b_temp);
        [x_temp, ~] = FP(ac, K, {}, b_temp, alpha, 20, 0.001);
        x = [x; x_temp]; 
        U = 0; y = 0; % dummy output
    else
        [U, y, ~] = SYTE_P1_V2(ac, K, B_na, alpha, 0, 4);
        x = 0; % dummy output
    end
end
time = toc(acc1_attr);
end



            
