function [w,b,out] = APG_LR_s(X,y,lam_w,lam_b,opts)
    %=============================================
    %
    % accelerated proximal gradient method for solving the logistic regression
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
    % Initialize variables following APG structure
    w = w0;
    b = b0;
    w_prev = w;
    b_prev = b;
    t = 1;
    t_prev = 1;
    hist_optErr = zeros(maxit, 1);
    timeout_seconds = 25; % Define timeout duration
    
    % Estimate step size (alpha) using Lipschitz constant
    L = 0.25 * norm(X,'fro')^2 / N + max(lam_w, lam_b);
    if L == 0
        L = 1;
    end
    alpha = 1/L;
    
    % Start timer
    timer_id = tic;
    timed_out = false;
    
    % Time tracking variables
    total_extrapolation_time = 0;
    total_gradient_time = 0;
    total_update_time = 0;
    
    for k = 1:maxit
        iter_start_time = tic;
        
        % Compute extrapolated point
        extrap_time = tic;
        beta = (t_prev - 1)/t;
        w_hat = w + beta*(w - w_prev);
        b_hat = b + beta*(b - b_prev);
        extrapolation_time = toc(extrap_time);
        total_extrapolation_time = total_extrapolation_time + extrapolation_time;
        
        % Compute gradient at extrapolated point
        grad_time = tic;
        z_hat = w_hat' * X + b_hat;
        sigma = 1./(1 + exp(-y .* z_hat)); % Sigmoid function
        
        grad_w_hat = X * ((sigma-1) .* y)' / N + lam_w * w_hat;
        grad_b_hat = sum((sigma-1) .* y) / N + lam_b * b_hat;
        gradient_time = toc(grad_time);
        total_gradient_time = total_gradient_time + gradient_time;
        
        % Update variables using gradient step
        update_time = tic;
        w_next = w_hat - alpha * grad_w_hat;
        b_next = b_hat - alpha * grad_b_hat;
        
        % Compute gradient at new point for optimality check
        z = w_next' * X + b_next;
        sigma_new = 1./(1 + exp(-y .* z));
        
        grad_w = X * ((sigma_new-1) .* y)' / N + lam_w * w_next;
        grad_b = sum((sigma_new-1) .* y) / N + lam_b * b_next;
        update_time = toc(update_time);
        total_update_time = total_update_time + update_time;
        
        % Calculate optimality error
        optErr = norm([grad_w; grad_b]);
        hist_optErr(k) = optErr;
        
        % Total iteration time
        iter_total_time = toc(iter_start_time);
        
        % Print timing breakdown for this iteration
        fprintf('Iter %3d: optErr=%.2e, time=%.3fs (extrap:%.3fs, grad:%.3fs, update:%.3fs)\n', ...
            k, optErr, iter_total_time, extrapolation_time, gradient_time, update_time);
        
        % Check stopping condition
        if optErr < tol
            hist_optErr = hist_optErr(1:k);
            break;
        end
        
        % Check for timeout
        elapsed_time = toc(timer_id);
        if elapsed_time > timeout_seconds
            timed_out = true;
            hist_optErr = hist_optErr(1:k);
            break;
        end
        
        % Update for next iteration
        t_next = (1 + sqrt(1 + 4*t^2))/2;
        
        w_prev = w;
        b_prev = b;
        w = w_next;
        b = b_next;
        t_prev = t;
        t = t_next;
    end
    
    % Record elapsed time
    elapsed_time = toc(timer_id);
    
    % Setup output structure
    out.hist_optErr = hist_optErr;
    out.iterations = k;
    out.elapsed_time = elapsed_time;
    out.timed_out = timed_out;
    out.time_breakdown = struct(...
        'extrapolation', total_extrapolation_time / k, ...
        'gradient', total_gradient_time / k, ...
        'update', total_update_time / k);
    
    % Print summary statistics
    fprintf('\nAPG Summary:\n');
    fprintf('Total time: %.3fs for %d iterations (%.3fs per iter)\n', elapsed_time, k, elapsed_time/k);
    fprintf('Time breakdown per iteration (avg): Extrapolation: %.3fs, Gradient: %.3fs, Update: %.3fs\n', ...
        total_extrapolation_time/k, total_gradient_time/k, total_update_time/k);
    
    if timed_out
        fprintf('APG algorithm timed out after %f seconds (%d iterations).\n', elapsed_time, k);
    end
end


