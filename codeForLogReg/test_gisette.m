%% classification on gisette

load gisette.mat;

% call your solver to have (w,b)
% you can tune the parameter lam_w, lam_b
% change the parameters if needed

Xtrain = Xtrain';
Xtest = Xtest';

[p,N] = size(Xtrain);

lam_w = 1e-4;
lam_b = 1e-4;
w_init = randn(p,1);
b_init = 0;

opts = [];
opts.tol = 1e-4;
opts.maxit = 1000;
opts.w0 = w_init;
opts.b0 = b_init;

m = 20; % number of iterates saved in memory by L-BFGS


% %%
% fprintf('Testing by student APG\n\n');


% t0 = tic;

% [w_s,b_s,out_s] = APG_LR_s(Xtrain,ytrain,lam_w,lam_b,opts);

% time = toc(t0);

% pred_y = sign(Xtest'*w_s + b_s);

% accu = sum(pred_y==ytest)/length(ytest);

% fprintf('Running time is %5.4f\n',time);
% fprintf('classification accuracy on testing data: %4.2f%%\n\n',accu*100);

% fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
% semilogy(out_s.hist_optErr,'b-','linewidth',2);
% xlabel('iteration');
% ylabel('Norm of Grad');
% title('gisette by APG of student');
% set(gca,'fontsize',14)
% print(fig,'-dpdf','gisette_APG_student')

% %%
% fprintf('Testing by student L-BFGS\n\n');

% t0 = tic;

% [w_s,b_s,out_s] = LBFGS_LR_s(Xtrain,ytrain,lam_w,lam_b,m,opts);

% time = toc(t0);

% pred_y = sign(Xtest'*w_s + b_s);

% accu = sum(pred_y==ytest)/length(ytest);

% fprintf('Running time is %5.4f\n',time);
% fprintf('classification accuracy on testing data: %4.2f%%\n\n',accu*100);

% fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
% semilogy(out_s.hist_optErr,'b-','linewidth',2);
% xlabel('iteration');
% ylabel('Norm of Grad');
% title('gisette by L-BFGS of student');
% set(gca,'fontsize',14)
% print(fig,'-dpdf','gisette_LBFGS_student')


%%
fprintf('Testing by instructor APG\n\n');

t0 = tic;

[w_p,b_p,out_p] = APG_LR_p(Xtrain,ytrain,lam_w,lam_b,opts);

time = toc(t0);

pred_y = sign(Xtest'*w_p + b_p);

accu = sum(pred_y==ytest)/length(ytest);

fprintf('Running time is %5.4f\n',time);
fprintf('classification accuracy on testing data: %4.2f%%\n\n',accu*100);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(out_p.hist_optErr,'b-','linewidth',2);
xlabel('iteration');
ylabel('Norm of Grad');
title('gisette by APG of instructor');
set(gca,'fontsize',14)
print(fig,'-dpdf','gisette_APG_instructor')

%%
fprintf('Testing by instructor L-BFGS\n\n');

t0 = tic;

[w_p,b_p,out_p] = LBFGS_LR_p(Xtrain,ytrain,lam_w,lam_b,m,opts);

time = toc(t0);

pred_y = sign(Xtest'*w_p + b_p);

accu = sum(pred_y==ytest)/length(ytest);

fprintf('Running time is %5.4f\n',time);
fprintf('classification accuracy on testing data: %4.2f%%\n\n',accu*100);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(out_p.hist_optErr,'b-','linewidth',2);
xlabel('iteration');
ylabel('Norm of Grad');
title('gisette by L-BFGS of instructor');
set(gca,'fontsize',14)
print(fig,'-dpdf','gisette_LBFGS_instructor')
