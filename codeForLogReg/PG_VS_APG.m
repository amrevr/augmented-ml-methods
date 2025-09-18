mu = 1e-3;
f = @(x) 0.5*mu*x^2 -log(1+x);
f1 = @(x) mu*x - 1./(1+x);

xinit = 10;
tol = 1e-6;

%% Proximal Gradient
% x = xinit;
% alpha = 0.5;
% k = 0;
% while abs(f1(x)) > tol
%     fprintf('k = %d, abs_grad = %5.2e\n', k, abs(f1(x)));
%     x = x - alpha * f1(x);
%     k = k + 1;   
% end
% 
% fprintf('k = %d, abs_grad = %5.2e\n\n', k, abs(f1(x)));

%% Accelerated Proximal Gradient
x0 = xinit;
y = x0;
x = x0;
k = 0;
t0 = 1;

while abs(f1(x)) > tol
    fprintf('k = %d, abs_grad = %5.2e\n', k, abs(f1(x)));
    x = y - alpha * f1(y);
    k = k + 1;
    t = (1+sqrt(1+4*t0^2)) / 2;
    w = (t0-1)/t;
    y = x + w*(x - x0);
    x0 = x; t0 = t;
end

fprintf('k = %d, abs_grad = %5.2e\n\n', k, abs(f1(x)));
