function [w,b,out] = LBFGS_LR_s(X,y,lam_w,lam_b,m,opts)
%=============================================
%
% limited-memory BFGS for solving the logistic regression
% min_{w,b} 1/N * sum_{i=1}^N log( 1+exp(-y(i)*(w'*X(:,i) + b)) ) +
% .5*lam_w*||w||^2 + .5*lam_b*b^2
%
%===============================================
%
% ==============================================
% input:
%       X: training data, each column is a sample data
%       y: label vector
%       lam_w, lam_b: model parameters
%       m: the number of historical iterates to save in memory
%       opts.tol: stopping tolerance
%       opts.maxit: maximum number of outer iteration
%       opts.w0: initial w
%       opts.b0: initial b0
%
% output:
%       w: learned w
%       b: learned b
%       out.hist_optErr: historical violation to optimality condition

%% get size of problem: p is dimension; N is number of data pts
[p,N] = size(X);

%% set parameters
if isfield(opts,'tol')        tol = opts.tol;           else tol = 1e-4;       end
if isfield(opts,'maxit')      maxit = opts.maxit;       else maxit = 500;      end
if isfield(opts,'w0')         w0 = opts.w0;             else w0 = zeros(p,1);  end
if isfield(opts,'b0')         b0 = opts.b0;             else b0 = 0;           end

%% main iterations
w = w0;
b = b0;
hist_optErr = zeros(maxit, 1);
timeout_seconds = 25; % Define timeout duration

% Initialize storage for L-BFGS
s = cell(1,m); % Storage for variable differences
y_hist = cell(1,m); % Storage for gradient differences
rho = zeros(1,m);
k = 1; % Iteration counter
mem_start = 1; % Index of the oldest element in memory
mem_end = 0; % Index of the newest element in memory
mem_size = 0; % Current size of memory

% Start timer
timer_id = tic;
timed_out = false;

% Initial function evaluation and gradient
z = w' * X + b;
sigma = 1./(1 + exp(-y .* z));
grad_w = X * ((sigma-1) .* y)' / N + lam_w * w;
grad_b = sum((sigma-1) .* y) / N + lam_b * b;
g = [grad_w; grad_b]; % Combined gradient

% Initial optimality error
optErr = norm(g);
hist_optErr(1) = optErr;

if optErr < tol
    % Already optimal, return
    out.hist_optErr = hist_optErr(1);
    out.iterations = 1;
    out.elapsed_time = toc(timer_id);
    out.timed_out = false;
    return;
end

% Main L-BFGS iteration
total_direction_time = 0;
total_linesearch_time = 0;
total_gradient_time = 0;
total_memory_time = 0;

for k = 1:maxit
    iter_start_time = tic;
    
    % Direction finding using two-loop recursion
    dir_time = tic;
    q = g;
    alpha = zeros(1,m);
    
    % First loop
    for i = 0:mem_size-1
        idx = mod(mem_end - i - 1, m) + 1;
        alpha(idx) = rho(idx) * dot(s{idx}, q);
        q = q - alpha(idx) * y_hist{idx};
    end
    
    % Initial Hessian approximation
    if mem_size > 0
        idx = mem_end;
        gamma = dot(s{idx}, y_hist{idx}) / dot(y_hist{idx}, y_hist{idx});
        H0 = gamma;
    else
        % Use identity as initial Hessian
        H0 = 1.0;
    end
    
    r = H0 * q;
    
    % Second loop
    for i = 0:mem_size-1
        idx = mod(mem_start + i - 1, m) + 1;
        beta = rho(idx) * dot(y_hist{idx}, r);
        r = r + s{idx} * (alpha(idx) - beta);
    end
    
    % Search direction (negative because we're minimizing)
    d = -r;
    direction_time = toc(dir_time);
    total_direction_time = total_direction_time + direction_time;
    
    % Line search for step size (simple backtracking)
    line_time = tic;
    alpha_step = 1.0;
    c = 1e-4; % Parameter for sufficient decrease condition
    
    % Current function value
    z = w' * X + b;
    sigma = 1./(1 + exp(-y .* z));
    f = sum(log(1 + exp(-y .* z))) / N + 0.5 * lam_w * sum(w.^2) + 0.5 * lam_b * b^2;
    
    ls_iters = 0;
    while true
        % Trial step
        w_new = w + alpha_step * d(1:p);
        b_new = b + alpha_step * d(end);
        
        % New function value
        z_new = w_new' * X + b_new;
        sigma_new = 1./(1 + exp(-y .* z_new));
        f_new = sum(log(1 + exp(-y .* z_new))) / N + 0.5 * lam_w * sum(w_new.^2) + 0.5 * lam_b * b_new^2;
        
        ls_iters = ls_iters + 1;
        % Check Armijo condition
        if f_new <= f + c * alpha_step * dot(g, d)
            break;
        end
        
        % Reduce step size
        alpha_step = alpha_step * 0.5;
        
        % Avoid too small step size
        if alpha_step < 1e-10
            alpha_step = 1e-10;
            break;
        end
    end
    linesearch_time = toc(line_time);
    total_linesearch_time = total_linesearch_time + linesearch_time;
    
    % Store old variables and gradient
    w_old = w;
    b_old = b;
    g_old = g;
    
    % Update variables
    w = w + alpha_step * d(1:p);
    b = b + alpha_step * d(end);
    
    % New gradient
    grad_time = tic;
    z = w' * X + b;
    sigma = 1./(1 + exp(-y .* z));
    grad_w = X * ((sigma-1) .* y)' / N + lam_w * w;
    grad_b = sum((sigma-1) .* y) / N + lam_b * b;
    g = [grad_w; grad_b];
    gradient_time = toc(grad_time);
    total_gradient_time = total_gradient_time + gradient_time;
    
    % Update L-BFGS memory
    mem_time = tic;
    s_k = [w - w_old; b - b_old]; % Variable difference
    y_k = g - g_old; % Gradient difference
    
    % Skip update if y_k'*s_k is too small (indicates poor curvature information)
    if dot(y_k, s_k) > 1e-10
        if mem_size < m
            % Memory not full yet
            mem_end = mem_end + 1;
            mem_size = mem_size + 1;
        else
            % Memory full, replace oldest
            mem_start = mod(mem_start, m) + 1;
            mem_end = mod(mem_end, m) + 1;
        end
        
        s{mem_end} = s_k;
        y_hist{mem_end} = y_k;
        rho(mem_end) = 1 / dot(y_k, s_k);
    end
    memory_time = toc(mem_time);
    total_memory_time = total_memory_time + memory_time;
    
    % Calculate optimality error
    optErr = norm(g);
    hist_optErr(k+1) = optErr;
    
    % Total iteration time
    iter_total_time = toc(iter_start_time);
    
    % Print timing breakdown for this iteration
    fprintf('Iter %3d: optErr=%.2e, time=%.3fs (dir:%.3fs, ls:%d steps/%.3fs, grad:%.3fs, mem:%.3fs)\n', ...
        k, optErr, iter_total_time, direction_time, ls_iters, linesearch_time, gradient_time, memory_time);
    
    % Check stopping condition
    if optErr < tol
        hist_optErr = hist_optErr(1:k+1);
        break;
    end
    
    % Check for timeout
    elapsed_time = toc(timer_id);
    if elapsed_time > timeout_seconds
        timed_out = true;
        hist_optErr = hist_optErr(1:k+1);
        break;
    end
end

% Record elapsed time
elapsed_time = toc(timer_id);

% Setup output structure
out.hist_optErr = hist_optErr(1:k+1);
out.iterations = k+1;
out.elapsed_time = elapsed_time;
out.timed_out = timed_out;
out.time_breakdown = struct(...
    'direction', total_direction_time / k, ...
    'linesearch', total_linesearch_time / k, ...
    'gradient', total_gradient_time / k, ...
    'memory', total_memory_time / k);

% Print summary statistics
fprintf('\nL-BFGS Summary:\n');
fprintf('Total time: %.3fs for %d iterations (%.3fs per iter)\n', elapsed_time, k, elapsed_time/k);
fprintf('Time breakdown per iteration (avg): Direction: %.3fs, Line search: %.3fs, Gradient: %.3fs, Memory update: %.3fs\n', ...
    total_direction_time/k, total_linesearch_time/k, total_gradient_time/k, total_memory_time/k);

if timed_out
    fprintf('L-BFGS algorithm timed out after %f seconds (%d iterations).\n', elapsed_time, k);
end

end


