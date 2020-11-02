function [x] = calculate_x(U, y, l, b_uni, ax1, ax2, ax3)
%% Efficiently calcualting long vector x from U and y

%% Faster method !
if b_uni == 1
    r1 = size(U{1,1}, 2); r2 = size(U{2,1}, 2);r3 = size(U{3,1}, 2);
    Y = reshape(y, r3, r1*r2);
    X = U{3,1} * Y;
    X = X * kron(U{1,1}, U{2,1})';
    x = reshape(X, ax1*ax2*ax3, 1);
else
    x = sparse(ax1*ax2*ax3, 1);
    for i = 1 : size(U, 1)
        Y = reshape(y{i, 1}, l, l^2);
        X = U{i,1}{3,1} * Y;
        X = X * kron(U{i,1}{1,1}, U{i,1}{2,1})';
        x0 = sparse(reshape(X, ax1*ax2*ax3, 1));
        x = x + x0;
    end
end

% when b_uni = 1, U is a cell, y is matrix (vector);
% when b_uni = 0, U, y are cells;

%% SLOW METHOD !
% inter1 = 1;
% inter2 = 1;
% inter3 = 1;
% x = [];
% x0 = [];
% if b_uni == 1
%     parfor i = 1:inter1:ax1
%         for j = 1:inter2:ax2
%             temp = kron(U{1,1}(i:i-1+inter1, :), U{2,1}(j:j-1+inter2, :));    
%             for k = 1:inter3:ax3    
%                 x = [x ; kron(temp, U{3,1}(k:k-1+inter3, :)) * y];
%             end
%         end
%     end
% else
%     parfor l = 1:size(U, 1)
%         for i = 1:inter1:ax1
%             for j = 1:inter2:ax2
%                 temp = kron(U{l,1}{1,1}(i:i-1+inter1, :), U{l,1}{2,1}(j:j-1+inter2, :));
%                 for k = 1:inter3:ax3
%                     x0 = [x0 ; kron(temp, U{3,1}(k:k-1+inter3, :)) * y];
%                 end
%             end
%         end
%         x = x0 + x;
%         x0 = zeros(ax1*ax2*ax3, 1);
%     end
% end

end
