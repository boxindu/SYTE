function [f,g] = subspace(R0)
%% Compute the subspace vectors f, g based on residual R0.
s1 = sum(abs(R0));
s2 = sum(abs(R0), 2);
[~,i1] = sort(s1, 'descend');
[~,i2] = sort(s2, 'descend');
if norm(R0,1) >= norm(R0,Inf)
    f = R0(:, i1(1));
    g = (R0'*f)/(norm(f,2).^2);
else
    g = R0(i2(1), :);
    g=g';
    f = (R0*g)/(norm(g,2).^2);
end

end