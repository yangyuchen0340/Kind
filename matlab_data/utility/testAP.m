function [FinalX, fv, gfv, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times, dist] = testAP(Uk,init,tol,maxit,idisp)
    [n,p] = size(Uk);
    fhandle = @(x)f(x, Uk);
    gfhandle = @(x)gf(x, Uk);
    Hesshandle = @(x, eta)Hess(x, eta, Uk);

    SolverParams.method = 'LRBFGS';
    % SolverParams.IsCheckParams = 1;
    SolverParams.DEBUG = idisp;
    % SolverParams.Min_Iteration = 100;
    % SolverParams.IsCheckGradHess = 1;
    SolverParams.Max_Iteration = maxit;
    SolverParams.OutputGap = 30;
    %SolverParams.IsStopped = @IsStopped;
    SolverParams.Stop_Criterion = 0;
    SolverParams.Tolerance = tol;
    SolverParams.LineSearch_LS = 0;
    %SolverParams.LinesearchInput = @LinesearchInput;

    % ManiParams.IsCheckParams = 1;
    ManiParams.name = 'OrthGroup';
    ManiParams.n = p;
    ManiParams.ParamSet = 1;
    HasHHR = 0;

    initialX.main = init;
    [FinalX, fv, gfv, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times] = DriverOPT(fhandle, gfhandle, Hesshandle, SolverParams, ManiParams, HasHHR, initialX);
    %[FinalX, fv, gfv, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times, dist] = DriverOPT(fhandle, gfhandle, Hesshandle, SolverParams, ManiParams, HasHHR, initialX, FinalX);
end

% function output = LinesearchInput(x, eta, t0, s0)
%     output = 1;
% end

function output = IsStopped(x, gf, f, ngf, ngf0)
    output = ngf / ngf0 < 1e-5;
end

function [output, x] = f(x, Uk)
output = 0.5*sum(sum(min(Uk*x.main,0).^2));
end

function [output, x] = gf(x, Uk)
output.main = x.main-Uk'*max(Uk*x.main,0);
end

function [output, x] = Hess(x, eta, Uk)
output.main = eta;
end
