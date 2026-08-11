


clear
clc
close all



% set random seed

rng(358655)




% PROBLEM 5, MODIFIED NEWTON AND TRUNCATED NEWTON

disp("PROBLEM 5")


% function, gradient and Hessian


f = @(x) problem_5(x);
gradf = @(x) get_gradient_5(x);
Hessf = @(x) get_hessian_5(x);


function g = get_gradient_5(x)
    [~, g] = problem_5(x);
end

function H = get_hessian_5(x)
    [~, ~, H] = problem_5(x);
end


% fine tuned parameters
% vedere se differenziare 

kmax = 1000; %bo 5000
tolgrad =  1e-8; %bo
c1 = 1e-4;
rho = 0.8;
btmax = 50;
beta = 1e-3; % nelle note 
jmax_M = 40; %bo
jmax_T= 500; %bo

eta =  @(x)  min(0.5, x); %quadratic
%eta =  @(x)  min(0.5, sqrt(x)); %superlinear






dims = [2, 1e3, 1e4, 1e5];

% warmup
x_warmup = -ones(2,1);
modified_newton_bcktrck(x_warmup,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, beta, jmax_M);
truncated_newton_bcktrck(x_warmup,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, jmax_T, eta);



for n = dims
    disp(n)
    start_points = [- ones(n,1), unifrnd(-2, 0, n,1), unifrnd(-2, 0, n,1), unifrnd(-2, 0, n,1), unifrnd(-2, 0, n,1), unifrnd(-2, 0, n,1)]; % deignated starting point and 5 randomly generated
    
    if n == 2
        xBigSeq_M = cell(length(start_points),1);
        xBigSeq_T = cell(length(start_points),1);
    end


    num = 1;
    for x0 = start_points
        
        % MODIFIED
        tic
        [xk_M,fk_M,gradfk_norm_M,k_M,xseq_M, btseq_M, flag_multi_M] = modified_newton_bcktrck(x0,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, beta, jmax_M);
        time_M = toc;

        % TRUNCATED
        tic
        [xk_T,fk_T,gradfk_norm_T,k_T,xseq_T, btseq_T, flag_multi_T] = truncated_newton_bcktrck(x0,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, jmax_T, eta);
        time_T = toc;

        if n==2
            xBigSeq_M{num} = xseq_M;
            xBigSeq_T{num} = xseq_T;
        end

        

        % EXPERIMENTAL RATE MODIFIED
        if k_M >= 3
            x_k   = xseq_M(:, 4);
            x_k1  = xseq_M(:, 3); 
            x_k2  = xseq_M(:, 2); 
            x_k3  = xseq_M(:, 1); 
            
            e_k   = norm(x_k - x_k1);
            e_k1  = norm(x_k1 - x_k2);
            e_k2  = norm(x_k2 - x_k3);
            
            % experimental rate
            rate_exp_M = log((e_k + eps) / (e_k1 + eps)) / log((e_k1 + eps) / (e_k2 + eps));
        else
            rate_exp_M = NaN;
        end



         % EXPERIMENTAL RATE TRUNCATED
        if k_T >= 3
            x_k   = xseq_T(:, 4);
            x_k1  = xseq_T(:, 3); 
            x_k2  = xseq_T(:, 2); 
            x_k3  = xseq_T(:, 1); 
            
            e_k   = norm(x_k - x_k1);
            e_k1  = norm(x_k1 - x_k2);
            e_k2  = norm(x_k2 - x_k3);
            
            % experimental rate
            rate_exp_T = log((e_k + eps) / (e_k1 + eps)) / log((e_k1 + eps) / (e_k2 + eps));
        else
            rate_exp_T = NaN;
        end
            

        % MODIFIED
        fprintf('  Modified  | Pt %d | Grad norm: %.2e | Iters: %d/%d | Flag: %d | Rate: %.2f | Time: %.4fs\n', ...
                num, gradfk_norm_M, k_M, kmax, flag_multi_M, rate_exp_M, time_M);

        % TRUNCATED
        fprintf('  Truncated | Pt %d | Grad norm: %.2e | Iters: %d/%d | Flag: %d | Rate: %.2f | Time: %.4fs\n\n', ...
                num, gradfk_norm_T, k_T, kmax, flag_multi_T, rate_exp_T, time_T);

        

        num = num+1;
    end

    fprintf('\n');

end

% figures


% TOP VIEW MODIFIED
figure;
[X, Y] = meshgrid(linspace(-6, 2, 500), linspace(-3, 5, 500));
Z = zeros(size(X));

for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        Z(i,j) = f([X(i,j); Y(i,j)]); 
    end
end


% better levels
z_min = min(Z(:)); 
z_max = max(Z(:));
levels = logspace(log10(z_min + 0.1), log10(z_max), 80) - 0.1; 

contour(X, Y, Z, levels); 
hold on;

plot(xBigSeq_M{1}(1,:), xBigSeq_M{1}(2,:), 'r.-', 'DisplayName', 'Start 1');
plot(xBigSeq_M{2}(1,:), xBigSeq_M{2}(2,:), 'b.-', 'DisplayName', 'Start 2');
plot(xBigSeq_M{3}(1,:), xBigSeq_M{3}(2,:), 'g.-', 'DisplayName', 'Start 3');
plot(xBigSeq_M{4}(1,:), xBigSeq_M{4}(2,:), 'm.-', 'DisplayName', 'Start 4');
plot(xBigSeq_M{5}(1,:), xBigSeq_M{5}(2,:),'c.-', 'DisplayName', 'Start 5');
plot(xBigSeq_M{6}(1,:), xBigSeq_M{6}(2,:), 'k.-', 'DisplayName', 'Start 6');

title('Top view of the function and sequence paths modified 5) (n=2)');
xlabel('x_1');
ylabel('x_2');
legend('show');
xlim([-2.5,1]);
ylim([-2.5, 0.5]);




% TOP VIEW TRUNCATED
figure;
[X, Y] = meshgrid(linspace(-6, 2, 500), linspace(-3, 5, 500));
Z = zeros(size(X));

for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        Z(i,j) = f([X(i,j); Y(i,j)]); 
    end
end

% better levels
z_min = min(Z(:)); 
z_max = max(Z(:));
levels = logspace(log10(z_min + 0.1), log10(z_max), 80) - 0.1; 

contour(X, Y, Z, levels); 
hold on;


plot(xBigSeq_T{1}(1,:), xBigSeq_T{1}(2,:), 'r.-', 'DisplayName', 'Start 1');
plot(xBigSeq_T{2}(1,:), xBigSeq_T{2}(2,:), 'b.-', 'DisplayName', 'Start 2');
plot(xBigSeq_T{3}(1,:), xBigSeq_T{3}(2,:), 'g.-', 'DisplayName', 'Start 3');
plot(xBigSeq_T{4}(1,:), xBigSeq_T{4}(2,:), 'm.-', 'DisplayName', 'Start 4');
plot(xBigSeq_T{5}(1,:), xBigSeq_T{5}(2,:),'c.-', 'DisplayName', 'Start 5');
plot(xBigSeq_T{6}(1,:), xBigSeq_T{6}(2,:), 'k.-', 'DisplayName', 'Start 6');

title('Top view of the function and sequence paths (truncated 5) (n=2)');
xlabel('x_1');
ylabel('x_2');
legend('show');
xlim([-2.5,1]);
ylim([-2.5, 0.5]);


%%

% PROBLEM 16, MODIFIED NEWTON AND TRUNCATED NEWTON

disp("PROBLEM 16")


% function, gradient and Hessian


f = @(x) problem_16(x);
gradf = @(x) get_gradient_16(x);
Hessf = @(x) get_hessian_16(x);

function g = get_gradient_16(x)
    [~, g] = problem_16(x);
end
function H = get_hessian_16(x)
    [~, ~, H] = problem_16(x);
end




% fine tuned parameters
% vedere se differenziare 

kmax = 1000; %bo 5000
tolgrad =  1e-8; %bo
c1 = 1e-4;
rho = 0.8;
btmax = 50;
%beta = 1e-3; % nelle note 
beta = 0.5;
jmax_M = 40; %bo
jmax_T= 500; %bo

eta =  @(x)  min(0.5, x); %quadratic
%eta =  @(x)  min(0.5, sqrt(x)); %superlinear





dims = [2, 1e3, 1e4, 1e5];

% warmup
x_warmup = -ones(2,1);
modified_newton_bcktrck(x_warmup,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, beta, jmax_M);
truncated_newton_bcktrck(x_warmup,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, jmax_T, eta);



for n = dims
    disp(n)

    start_points = [ones(n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1)]; % deignated starting point and 5 randomly generated

    
    if n == 2
        xBigSeq_M = cell(length(start_points),1);
        xBigSeq_T = cell(length(start_points),1);
    end

    num = 1;
    for x0 = start_points
        
        % MODIFIED
        tic
        [xk_M,fk_M,gradfk_norm_M,k_M,xseq_M, btseq_M, flag_multi_M] = modified_newton_bcktrck(x0,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, beta, jmax_M);
        time_M = toc;

        % TRUNCATED
        tic
        [xk_T,fk_T,gradfk_norm_T,k_T,xseq_T, btseq_T, flag_multi_T] = truncated_newton_bcktrck(x0,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, jmax_T, eta);
        time_T = toc;


        if n==2
            xBigSeq_M{num} = xseq_M;
            xBigSeq_T{num} = xseq_T;
        end


        

        % EXPERIMENTAL RATE MODIFIED
        if k_M >= 3
            x_k   = xseq_M(:, end);
            x_k1  = xseq_M(:, end-1); 
            x_k2  = xseq_M(:, end-2); 
            x_k3  = xseq_M(:, end-3); 
            
            e_k   = norm(x_k - x_k1);
            e_k1  = norm(x_k1 - x_k2);
            e_k2  = norm(x_k2 - x_k3);
            
            % experimental rate
            rate_exp_M = log((e_k + eps) / (e_k1 + eps)) / log((e_k1 + eps) / (e_k2 + eps));
        else
            rate_exp_M = NaN;
        end



         % EXPERIMENTAL RATE TRUNCATED
        if k_T >= 3
            x_k   = xseq_T(:, end);
            x_k1  = xseq_T(:, end-1); 
            x_k2  = xseq_T(:, end-2); 
            x_k3  = xseq_T(:, end-3); 
            
            e_k   = norm(x_k - x_k1);
            e_k1  = norm(x_k1 - x_k2);
            e_k2  = norm(x_k2 - x_k3);
            
            % experimental rate
            rate_exp_T = log((e_k + eps) / (e_k1 + eps)) / log((e_k1 + eps) / (e_k2 + eps));
        else
            rate_exp_T = NaN;
        end
            

        % MODIFIED
        fprintf('  Modified  | Pt %d | Grad norm: %.2e | Iters: %d/%d | Flag: %d | Rate: %.2f | Time: %.4fs\n', ...
                num, gradfk_norm_M, k_M, kmax, flag_multi_M, rate_exp_M, time_M);

        % TRUNCATED
        fprintf('  Truncated | Pt %d | Grad norm: %.2e | Iters: %d/%d | Flag: %d | Rate: %.2f | Time: %.4fs\n\n', ...
                num, gradfk_norm_T, k_T, kmax, flag_multi_T, rate_exp_T, time_T);

        

        num = num+1;
    end

    fprintf('\n');

end






%figures


% TOP VIEW MODIFIED
figure;
[X, Y] = meshgrid(linspace(-6, 2, 500), linspace(-3, 5, 500));
Z = zeros(size(X));

for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        Z(i,j) = f([X(i,j); Y(i,j)]); 
    end
end

contour(X, Y, Z, 50); % 50 level curves
hold on;

plot(xBigSeq_M{1}(1,:), xBigSeq_M{1}(2,:), 'r.-', 'DisplayName', 'Start 1');
plot(xBigSeq_M{2}(1,:), xBigSeq_M{2}(2,:), 'b.-', 'DisplayName', 'Start 2');
plot(xBigSeq_M{3}(1,:), xBigSeq_M{3}(2,:), 'g.-', 'DisplayName', 'Start 3');
plot(xBigSeq_M{4}(1,:), xBigSeq_M{4}(2,:), 'm.-', 'DisplayName', 'Start 4');
plot(xBigSeq_M{5}(1,:), xBigSeq_M{5}(2,:),'c.-', 'DisplayName', 'Start 5');
plot(xBigSeq_M{6}(1,:), xBigSeq_M{6}(2,:), 'k.-', 'DisplayName', 'Start 6');

title('Top view of the function and sequence paths (modified, 16) (n=2)');
xlabel('x_1');
ylabel('x_2');
legend('show');



% TOP VIEW TRUNCATED
figure;
[X, Y] = meshgrid(linspace(-6, 2, 500), linspace(-3, 5, 500));
Z = zeros(size(X));

for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        Z(i,j) = f([X(i,j); Y(i,j)]); 
    end
end

contour(X, Y, Z, 50); % 50 level curves
hold on;

plot(xBigSeq_T{1}(1,:), xBigSeq_T{1}(2,:), 'r.-', 'DisplayName', 'Start 1');
plot(xBigSeq_T{2}(1,:), xBigSeq_T{2}(2,:), 'b.-', 'DisplayName', 'Start 2');
plot(xBigSeq_T{3}(1,:), xBigSeq_T{3}(2,:), 'g.-', 'DisplayName', 'Start 3');
plot(xBigSeq_T{4}(1,:), xBigSeq_T{4}(2,:), 'm.-', 'DisplayName', 'Start 4');
plot(xBigSeq_T{5}(1,:), xBigSeq_T{5}(2,:),'c.-', 'DisplayName', 'Start 5');
plot(xBigSeq_T{6}(1,:), xBigSeq_T{6}(2,:), 'k.-', 'DisplayName', 'Start 6');

title('Top view of the function and sequence paths (truncated, 16) (n=2)');
xlabel('x_1');
ylabel('x_2');
legend('show');





 %% es 9.4

 




c1 = 1e-4;
rho = 0.8;
btmax = 50;
x0 = [1, 0.5,0]; %alpha, phi, psi
kmax = 5000;
gradftol = 1e-4;

F = @(x) x(1)*sin(wave_xx* x(2) +x(3) ) - wave_yy;
%JF = @(x) x(1)*sin(wave_xx* x(2) +x(3) );

% wave_X = [wave_xx, ones(size(wave_xx, 1),1)];
% [wave_w,Fk, normFk,normgradfk, k, xseq, btseq] = gaussnewton(x0,F,JF,kmax,gradftol,c1, rho, btmax);
% wave_w %w and b



figure
plot(sort(lin_xx), true_line(sort(lin_xx)));
hold on
plot(sort(lin_xx), lin_w(2) + lin_w(1).*sort(lin_xx));
hold on
scatter((lin_xx), lin_yy);
hold off

figure
plot(sort(philin_xx), true_philine((sort(philin_xx))));
hold on
plot(sort(philin_xx), philin_w(2) + philin_w(1).*g(sort(philin_xx)));
hold on
scatter((philin_xx), philin_yy);
hold off


figure
plot(sort(wave_xx), true_wave((sort(wave_xx))));
hold on
scatter((wave_xx), wave_yy);

hold on
plot(sort(wave_xx), (wave_w(3) + wave_w(2).*(sort(wave_xx)))*wave_w(1));

hold off