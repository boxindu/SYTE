# SYTE
-- This is a MATLAB implementation of the algorithms proposed in the paper:
    Sylvester Tensor Equation for Multi-Way Association. 
-- Tested on MATLAB R2017b and MATLAB 2019a.
-- Requirements: MATLAB tensor toolbox (attched in the folder named tensor_toolbox-v3.1)
-- To use the tensor toolbox, go to the tensor_toolbox-v3.1 folder and run:
    addpath(pwd)
    savepath
-- The instructions of how to use and input/output information are detailed in each function.
-- One small sampled dataset of plain graph is provided for demo purpose. 
    To use, in the command window, one can run:
    load('3-small-douban.mat')
    [U1, y1, time1, ~] = SYTE_P1(AC, 3, ones(230*230*242, 1), 0.5, 1, 10, 0);
    [U2, y2, time2, ~] = SYTE_P2(AC, 3, 2, ones(230*230*242, 1), 0.5, 0);
    [U3, y3, ~] = SYTE_P1_V2(AC, 3, B, 0.5, 1, 4);
