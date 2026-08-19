
clear
clc
close all



% set random seed

rng(358655)


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

kmax_M = 1000; %bo 
kmax_T = 5000; %bo truncated raggiungeva kmax 1000 3
tolgrad_M =  1e-8; %bo
tolgrad_M_big = 1e-5; % raggiungeva 1000 it per la dim piu alta 6
tolgrad_T =  1e-6; %bo 5
c1 = 1e-4;
rho_M = 0.8;
rho_T = 0.5; % evitare flag 2 su truncated 2
btmax_M = 50;
btmax_T = 100;
%beta = 1e-3; % nelle note 
beta = 0.5; % evitare che vadano in altri minimi 1
jmax_M = 40; %bo
jmax_T= 50; %bo era 500  ci metteva troppo e non avevamo buoni risultati 7

eta =  @(x)  min(0.5, x); %quadratic 
%eta =  @(x)  min(0.5, sqrt(x)); %superlinear 4 ma non ha funzionato





%dims = [2, 1e3, 1e4, 1e5];
dims = [1e4];

% warmup
x_warmup = -ones(2,1);
modified_newton_bcktrck(x_warmup,f,gradf,Hessf,kmax_M,tolgrad_M,c1, rho_M, btmax_M, beta, jmax_M);
truncated_newton_bcktrck(x_warmup,f,gradf,Hessf,kmax_T,tolgrad_T,c1, rho_T, btmax_T, jmax_T, eta);


h_vec=[1e-4, 1e-8, 1e-12 ];
for h = h_vec    
    
    disp('Constant gradient and Hessian');
    disp(h);

    Hessf_const_handle = @(x) findiff_Hess_16(x, h);
    gradf_const_handle = @(x) findiff_grad_16(x, h); 
    for n = dims
        disp(n)
    
        start_points = [ones(n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1)]; % deignated starting point and 5 randomly generated
        
        % storing gradient norms
        gradBigSeq_M = cell(length(start_points), 1);
        gradBigSeq_T = cell(length(start_points), 1);
    
        
        if n == 2
            xBigSeq_M = cell(length(start_points),1);
            xBigSeq_T = cell(length(start_points),1);
        end
    
        % storing rates
        rateBigSeq_M = cell(length(start_points), 1);
        rateBigSeq_T = cell(length(start_points), 1);
    
        num = 1;
        for x0 = start_points
            
            % MODIFIED
    
            if n == 1e5
                tic
                [xk_M,fk_M,gradfk_norm_vec_M,k_M,xseq_M, btseq_M, flag_multi_M, err_seq_M] = modified_newton_bcktrck(x0,f,gradf_const_handle,Hessf_const_handle,kmax_M,tolgrad_M_big,c1, rho_M, btmax_M, beta, jmax_M);
                time_M = toc;
            else
                tic
                [xk_M,fk_M,gradfk_norm_vec_M,k_M,xseq_M, btseq_M, flag_multi_M, err_seq_M] = modified_newton_bcktrck(x0,f,gradf_const_handle,Hessf_const_handle,kmax_M,tolgrad_M,c1, rho_M, btmax_M, beta, jmax_M);
                time_M = toc;
            end
    
            % TRUNCATED
            tic
            [xk_T,fk_T,gradfk_norm_vec_T,k_T,xseq_T, btseq_T, flag_multi_T, err_seq_T] = truncated_newton_bcktrck(x0,f,gradf_const_handle,Hessf_const_handle,kmax_T,tolgrad_T,c1, rho_T, btmax_T, jmax_T, eta);
            time_T = toc;
    
    
            if n==2
                xBigSeq_M{num} = xseq_M;
                xBigSeq_T{num} = xseq_T;
            end
    
             % storing gradient norms
            gradBigSeq_M{num} = gradfk_norm_vec_M;
            gradBigSeq_T{num} = gradfk_norm_vec_T;
    
    
            % storing rates
            rateBigSeq_M{num} = get_rate_vector(err_seq_M);
            rateBigSeq_T{num} = get_rate_vector(err_seq_T);
            
    
    
            % EXPERIMENTAL RATE MODIFIED
            if ~isempty(rateBigSeq_M{num})
                rate_exp_M = rateBigSeq_M{num}(end);
            else
                rate_exp_M = NaN;
            end
    
            % EXPERIMENTAL RATE TRUNCATED
            if ~isempty(rateBigSeq_T{num})
                rate_exp_T = rateBigSeq_T{num}(end);
            else
                rate_exp_T = NaN; 
            end
            
                
            
    
            % MODIFIED
            fprintf('  Modified  | Pt %d | Grad norm: %.2e | Iters: %d/%d | Flag: %d | Rate: %.2f | Time: %.4fs\n', ...
                    num, gradfk_norm_vec_M(end), k_M, kmax_M, flag_multi_M, rate_exp_M, time_M);
    
            % TRUNCATED
            fprintf('  Truncated | Pt %d | Grad norm: %.2e | Iters: %d/%d | Flag: %d | Rate: %.2f | Time: %.4fs\n\n', ...
                    num, gradfk_norm_vec_T(end), k_T, kmax_T, flag_multi_T, rate_exp_T, time_T);
    
            
    
            num = num+1;
        end
    
        fprintf('\n');
    
        % FIGURES
    
        
        % GRADIENT NORMS MODIFIED
        
        figure;
        
        semilogy(0:length(gradBigSeq_M{1})-1, gradBigSeq_M{1}, 'r.-', 'DisplayName', 'Start 1');
        hold on;
        semilogy(0:length(gradBigSeq_M{2})-1, gradBigSeq_M{2}, 'b.-', 'DisplayName', 'Start 2');
        semilogy(0:length(gradBigSeq_M{3})-1, gradBigSeq_M{3}, 'g.-', 'DisplayName', 'Start 3');
        semilogy(0:length(gradBigSeq_M{4})-1, gradBigSeq_M{4}, 'm.-', 'DisplayName', 'Start 4');
        semilogy(0:length(gradBigSeq_M{5})-1, gradBigSeq_M{5}, 'c.-', 'DisplayName', 'Start 5');
        semilogy(0:length(gradBigSeq_M{6})-1, gradBigSeq_M{6}, 'k.-', 'DisplayName', 'Start 6');
        
        title(sprintf('Gradient norm, problem 16, Modified (n=%d)', n));

        xlabel('Iterations (k)');
        ylabel('||\nabla f(x_k)|| (Log Scale)');
        grid on;
        legend('show');
    
        
        
        % GRADIENT NORMS TRUNCATED
        
        figure;
        
        semilogy(0:length(gradBigSeq_T{1})-1, gradBigSeq_T{1}, 'r.-', 'DisplayName', 'Start 1');
        hold on;
        semilogy(0:length(gradBigSeq_T{2})-1, gradBigSeq_T{2}, 'b.-', 'DisplayName', 'Start 2');
        semilogy(0:length(gradBigSeq_T{3})-1, gradBigSeq_T{3}, 'g.-', 'DisplayName', 'Start 3');
        semilogy(0:length(gradBigSeq_T{4})-1, gradBigSeq_T{4}, 'm.-', 'DisplayName', 'Start 4');
        semilogy(0:length(gradBigSeq_T{5})-1, gradBigSeq_T{5}, 'c.-', 'DisplayName', 'Start 5');
        semilogy(0:length(gradBigSeq_T{6})-1, gradBigSeq_T{6}, 'k.-', 'DisplayName', 'Start 6');
        
        title(sprintf('Gradient norm, problem 16, Truncated (n=%d)', n));
        xlabel('Iterations (k)');
        ylabel('||\nabla f(x_k)|| (Log Scale)');
        grid on;
        legend('show');
        
    
    
    
        % EXPERIMENTAL RATES OF CONVERGENCE MODIFIED
    
        figure;
        plot(0:length(rateBigSeq_M{1})-1, rateBigSeq_M{1}, 'r.-', 'DisplayName', 'Start 1'); hold on;
        plot(0:length(rateBigSeq_M{2})-1, rateBigSeq_M{2}, 'b.-', 'DisplayName', 'Start 2');
        plot(0:length(rateBigSeq_M{3})-1, rateBigSeq_M{3}, 'g.-', 'DisplayName', 'Start 3');
        plot(0:length(rateBigSeq_M{4})-1, rateBigSeq_M{4}, 'm.-', 'DisplayName', 'Start 4');
        plot(0:length(rateBigSeq_M{5})-1, rateBigSeq_M{5}, 'c.-', 'DisplayName', 'Start 5');
        plot(0:length(rateBigSeq_M{6})-1, rateBigSeq_M{6}, 'k.-', 'DisplayName', 'Start 6');
        
        title(sprintf('Experimental Rate, problem 16, Modified Newton (n = %d)', n));

        xlabel('Iterations (k)');
        ylabel('Experimental Rate (p_k)');
        ylim([0, 3]); 
        grid on;
        legend('show');
    
        % EXPERIMENTAL RATES OF CONVERGENCE TRUNCATED
    
        figure;
        plot(0:length(rateBigSeq_T{1})-1, rateBigSeq_T{1}, 'r.-', 'DisplayName', 'Start 1'); hold on;
        plot(0:length(rateBigSeq_T{2})-1, rateBigSeq_T{2}, 'b.-', 'DisplayName', 'Start 2');
        plot(0:length(rateBigSeq_T{3})-1, rateBigSeq_T{3}, 'g.-', 'DisplayName', 'Start 3');
        plot(0:length(rateBigSeq_T{4})-1, rateBigSeq_T{4}, 'm.-', 'DisplayName', 'Start 4');
        plot(0:length(rateBigSeq_T{5})-1, rateBigSeq_T{5}, 'c.-', 'DisplayName', 'Start 5');
        plot(0:length(rateBigSeq_T{6})-1, rateBigSeq_T{6}, 'k.-', 'DisplayName', 'Start 6');
        

        title(sprintf('Experimental Rate, problem 16, Truncated Newton (n = %d)', n));

        xlabel('Iterations (k)');
        ylabel('Experimental Rate (p_k)');
        ylim([0, 3]); 
        grid on;
        legend('show');
        
        
        
    
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
    
    % better levels
    z_min = min(real(Z(:))); 
    z_max = max(real(Z(:)));
    levels = logspace(real(log10(z_min + 0.1)), real(log10(z_max)), 80) - 0.1; 

    contour(X, Y, real(Z), real(levels), 'LineColor', [0.7 0.7 0.7]);
    hold on;
    
    c1 = [0.000, 0.447, 0.741]; 
    c2 = [0.850, 0.325, 0.098]; 
    c3 = [0.466, 0.674, 0.188]; 
    c4 = [0.494, 0.184, 0.556]; 
    c5 = [0.929, 0.694, 0.125]; 
    c6 = [0.850, 0.000, 0.150];
    
    lw = 1.5; 
    ms = 12;  
    
    plot(xBigSeq_M{1}(1,:), xBigSeq_M{1}(2,:), '.-', 'Color', c1, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 1');
    plot(xBigSeq_M{2}(1,:), xBigSeq_M{2}(2,:), '.-', 'Color', c2, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 2');
    plot(xBigSeq_M{3}(1,:), xBigSeq_M{3}(2,:), '.-', 'Color', c3, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 3');
    plot(xBigSeq_M{4}(1,:), xBigSeq_M{4}(2,:), '.-', 'Color', c4, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 4');
    plot(xBigSeq_M{5}(1,:), xBigSeq_M{5}(2,:), '.-', 'Color', c5, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 5');
    plot(xBigSeq_M{6}(1,:), xBigSeq_M{6}(2,:), '.-', 'Color', c6, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 6');
    

    title('Top view of the function and sequence paths, problem 16, Modified (n=2)');

    xlabel('x_1');
    ylabel('x_2');
    legend('show');
    xlim([-2.5, 1]);
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
    z_min = min(real(Z(:))); 
    z_max = max(real(Z(:)));
    levels = logspace(real(log10(z_min + 0.1)), real(log10(z_max)), 80) - 0.1; 

    contour(X, Y, real(Z), real(levels), 'LineColor', [0.7 0.7 0.7]);
    hold on;
    
    c1 = [0.000, 0.447, 0.741]; 
    c2 = [0.850, 0.325, 0.098]; 
    c3 = [0.466, 0.674, 0.188]; 
    c4 = [0.494, 0.184, 0.556]; 
    c5 = [0.929, 0.694, 0.125]; 
    c6 = [0.850, 0.000, 0.150];
    
    lw = 1.5; 
    ms = 12;  
    
    plot(xBigSeq_T{1}(1,:), xBigSeq_T{1}(2,:), '.-', 'Color', c1, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 1');
    plot(xBigSeq_T{2}(1,:), xBigSeq_T{2}(2,:), '.-', 'Color', c2, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 2');
    plot(xBigSeq_T{3}(1,:), xBigSeq_T{3}(2,:), '.-', 'Color', c3, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 3');
    plot(xBigSeq_T{4}(1,:), xBigSeq_T{4}(2,:), '.-', 'Color', c4, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 4');
    plot(xBigSeq_T{5}(1,:), xBigSeq_T{5}(2,:), '.-', 'Color', c5, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 5');
    plot(xBigSeq_T{6}(1,:), xBigSeq_T{6}(2,:), '.-', 'Color', c6, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 6');
    
    
    
    
    title('Top view of the function and sequence paths, problem 16, Truncated (n=2)');

    xlabel('x_1');
    ylabel('x_2');
    legend('show');
    xlim([-2.5,1]);
    ylim([-2.5, 0.5]);


    % variable Hessian
    Hessf_var_handle = @(x) findiff_Hess_16(x, h * max(abs(x), 1e-12));
    gradf_var_handle = @(x) findiff_grad_16(x, h * max(abs(x), 1e-12));
    
    disp('Variable gradient and Hessian');
    disp(h);

    for n = dims
        disp(n)
    
        start_points = [ones(n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1), unifrnd(0, 2, n,1)]; % deignated starting point and 5 randomly generated
        
        % storing gradient norms
        gradBigSeq_M = cell(length(start_points), 1);
        gradBigSeq_T = cell(length(start_points), 1);
    
        
        if n == 2
            xBigSeq_M = cell(length(start_points),1);
            xBigSeq_T = cell(length(start_points),1);
        end
    
        % storing rates
        rateBigSeq_M = cell(length(start_points), 1);
        rateBigSeq_T = cell(length(start_points), 1);
    
        num = 1;
        for x0 = start_points
            
            % MODIFIED
    
            if n == 1e5
                tic
                [xk_M,fk_M,gradfk_norm_vec_M,k_M,xseq_M, btseq_M, flag_multi_M, err_seq_M] = modified_newton_bcktrck(x0,f,gradf_var_handle,Hessf_var_handle,kmax_M,tolgrad_M_big,c1, rho_M, btmax_M, beta, jmax_M);
                time_M = toc;
            else
                tic
                [xk_M,fk_M,gradfk_norm_vec_M,k_M,xseq_M, btseq_M, flag_multi_M, err_seq_M] = modified_newton_bcktrck(x0,f,gradf_var_handle,Hessf_var_handle,kmax_M,tolgrad_M,c1, rho_M, btmax_M, beta, jmax_M);
                time_M = toc;
            end
    
            % TRUNCATED
            tic
            [xk_T,fk_T,gradfk_norm_vec_T,k_T,xseq_T, btseq_T, flag_multi_T, err_seq_T] = truncated_newton_bcktrck(x0,f,gradf_var_handle,Hessf_var_handle,kmax_T,tolgrad_T,c1, rho_T, btmax_T, jmax_T, eta);
            time_T = toc;
    
    
            if n==2
                xBigSeq_M{num} = xseq_M;
                xBigSeq_T{num} = xseq_T;
            end
    
             % storing gradient norms
            gradBigSeq_M{num} = gradfk_norm_vec_M;
            gradBigSeq_T{num} = gradfk_norm_vec_T;
    
    
            % storing rates
            rateBigSeq_M{num} = get_rate_vector(err_seq_M);
            rateBigSeq_T{num} = get_rate_vector(err_seq_T);
            
    
    
            % EXPERIMENTAL RATE MODIFIED
            if ~isempty(rateBigSeq_M{num})
                rate_exp_M = rateBigSeq_M{num}(end);
            else
                rate_exp_M = NaN;
            end
    
            % EXPERIMENTAL RATE TRUNCATED
            if ~isempty(rateBigSeq_T{num})
                rate_exp_T = rateBigSeq_T{num}(end);
            else
                rate_exp_T = NaN; 
            end
            
                
            
    
            % MODIFIED
            fprintf('  Modified  | Pt %d | Grad norm: %.2e | Iters: %d/%d | Flag: %d | Rate: %.2f | Time: %.4fs\n', ...
                    num, gradfk_norm_vec_M(end), k_M, kmax_M, flag_multi_M, rate_exp_M, time_M);
    
            % TRUNCATED
            fprintf('  Truncated | Pt %d | Grad norm: %.2e | Iters: %d/%d | Flag: %d | Rate: %.2f | Time: %.4fs\n\n', ...
                    num, gradfk_norm_vec_T(end), k_T, kmax_T, flag_multi_T, rate_exp_T, time_T);
    
            
    
            num = num+1;
        end
    
        fprintf('\n');
    
        % FIGURES
    
        
        % GRADIENT NORMS MODIFIED
        
        figure;
        
        semilogy(0:length(gradBigSeq_M{1})-1, gradBigSeq_M{1}, 'r.-', 'DisplayName', 'Start 1');
        hold on;
        semilogy(0:length(gradBigSeq_M{2})-1, gradBigSeq_M{2}, 'b.-', 'DisplayName', 'Start 2');
        semilogy(0:length(gradBigSeq_M{3})-1, gradBigSeq_M{3}, 'g.-', 'DisplayName', 'Start 3');
        semilogy(0:length(gradBigSeq_M{4})-1, gradBigSeq_M{4}, 'm.-', 'DisplayName', 'Start 4');
        semilogy(0:length(gradBigSeq_M{5})-1, gradBigSeq_M{5}, 'c.-', 'DisplayName', 'Start 5');
        semilogy(0:length(gradBigSeq_M{6})-1, gradBigSeq_M{6}, 'k.-', 'DisplayName', 'Start 6');
        
        title(sprintf('Gradient norm, problem 16, Modified (n=%d)', n));

        xlabel('Iterations (k)');
        ylabel('||\nabla f(x_k)|| (Log Scale)');
        grid on;
        legend('show');
    
        
        
        % GRADIENT NORMS TRUNCATED
        
        figure;
        
        semilogy(0:length(gradBigSeq_T{1})-1, gradBigSeq_T{1}, 'r.-', 'DisplayName', 'Start 1');
        hold on;
        semilogy(0:length(gradBigSeq_T{2})-1, gradBigSeq_T{2}, 'b.-', 'DisplayName', 'Start 2');
        semilogy(0:length(gradBigSeq_T{3})-1, gradBigSeq_T{3}, 'g.-', 'DisplayName', 'Start 3');
        semilogy(0:length(gradBigSeq_T{4})-1, gradBigSeq_T{4}, 'm.-', 'DisplayName', 'Start 4');
        semilogy(0:length(gradBigSeq_T{5})-1, gradBigSeq_T{5}, 'c.-', 'DisplayName', 'Start 5');
        semilogy(0:length(gradBigSeq_T{6})-1, gradBigSeq_T{6}, 'k.-', 'DisplayName', 'Start 6');
        
        title(sprintf('Gradient norm, problem 16, Truncated (n=%d)', n));

        xlabel('Iterations (k)');
        ylabel('||\nabla f(x_k)|| (Log Scale)');
        grid on;
        legend('show');
        
    
    
    
        % EXPERIMENTAL RATES OF CONVERGENCE MODIFIED
    
        figure;
        plot(0:length(rateBigSeq_M{1})-1, rateBigSeq_M{1}, 'r.-', 'DisplayName', 'Start 1'); hold on;
        plot(0:length(rateBigSeq_M{2})-1, rateBigSeq_M{2}, 'b.-', 'DisplayName', 'Start 2');
        plot(0:length(rateBigSeq_M{3})-1, rateBigSeq_M{3}, 'g.-', 'DisplayName', 'Start 3');
        plot(0:length(rateBigSeq_M{4})-1, rateBigSeq_M{4}, 'm.-', 'DisplayName', 'Start 4');
        plot(0:length(rateBigSeq_M{5})-1, rateBigSeq_M{5}, 'c.-', 'DisplayName', 'Start 5');
        plot(0:length(rateBigSeq_M{6})-1, rateBigSeq_M{6}, 'k.-', 'DisplayName', 'Start 6');
        
        title(sprintf('Experimental Rate, problem 16, Modified Newton (n = %d)', n));

        xlabel('Iterations (k)');
        ylabel('Experimental Rate (p_k)');
        ylim([0, 3]); 
        grid on;
        legend('show');
    
        % EXPERIMENTAL RATES OF CONVERGENCE TRUNCATED
    
        figure;
        plot(0:length(rateBigSeq_T{1})-1, rateBigSeq_T{1}, 'r.-', 'DisplayName', 'Start 1'); hold on;
        plot(0:length(rateBigSeq_T{2})-1, rateBigSeq_T{2}, 'b.-', 'DisplayName', 'Start 2');
        plot(0:length(rateBigSeq_T{3})-1, rateBigSeq_T{3}, 'g.-', 'DisplayName', 'Start 3');
        plot(0:length(rateBigSeq_T{4})-1, rateBigSeq_T{4}, 'm.-', 'DisplayName', 'Start 4');
        plot(0:length(rateBigSeq_T{5})-1, rateBigSeq_T{5}, 'c.-', 'DisplayName', 'Start 5');
        plot(0:length(rateBigSeq_T{6})-1, rateBigSeq_T{6}, 'k.-', 'DisplayName', 'Start 6');
        
        title(sprintf('Experimental Rate, problem 16, Truncated Newton (n = %d)', n));

        xlabel('Iterations (k)');
        ylabel('Experimental Rate (p_k)');
        ylim([0, 3]); 
        grid on;
        legend('show');
        
        
        
    
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
    
    % better levels
    z_min = min(real(Z(:))); 
    z_max = max(real(Z(:)));
    levels = logspace(real(log10(z_min + 0.1)), real(log10(z_max)), 80) - 0.1; 

    contour(X, Y, real(Z), real(levels), 'LineColor', [0.7 0.7 0.7]);
    hold on;
    
    c1 = [0.000, 0.447, 0.741]; 
    c2 = [0.850, 0.325, 0.098]; 
    c3 = [0.466, 0.674, 0.188]; 
    c4 = [0.494, 0.184, 0.556]; 
    c5 = [0.929, 0.694, 0.125]; 
    c6 = [0.850, 0.000, 0.150];
    
    lw = 1.5; 
    ms = 12;  
    
    plot(xBigSeq_M{1}(1,:), xBigSeq_M{1}(2,:), '.-', 'Color', c1, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 1');
    plot(xBigSeq_M{2}(1,:), xBigSeq_M{2}(2,:), '.-', 'Color', c2, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 2');
    plot(xBigSeq_M{3}(1,:), xBigSeq_M{3}(2,:), '.-', 'Color', c3, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 3');
    plot(xBigSeq_M{4}(1,:), xBigSeq_M{4}(2,:), '.-', 'Color', c4, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 4');
    plot(xBigSeq_M{5}(1,:), xBigSeq_M{5}(2,:), '.-', 'Color', c5, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 5');
    plot(xBigSeq_M{6}(1,:), xBigSeq_M{6}(2,:), '.-', 'Color', c6, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 6');
    
    title('Top view of the function and sequence paths, problem 16, Modified (n=2)');

    xlabel('x_1');
    ylabel('x_2');
    legend('show');
    xlim([-2.5, 1]);
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
    z_min = min(real(Z(:))); 
    z_max = max(real(Z(:)));
    levels = logspace(real(log10(z_min + 0.1)), real(log10(z_max)), 80) - 0.1; 

    contour(X, Y, real(Z), real(levels), 'LineColor', [0.7 0.7 0.7]);
    hold on;
    
    c1 = [0.000, 0.447, 0.741]; 
    c2 = [0.850, 0.325, 0.098]; 
    c3 = [0.466, 0.674, 0.188]; 
    c4 = [0.494, 0.184, 0.556]; 
    c5 = [0.929, 0.694, 0.125]; 
    c6 = [0.850, 0.000, 0.150];
    
    lw = 1.5; 
    ms = 12;  
    
    plot(xBigSeq_T{1}(1,:), xBigSeq_T{1}(2,:), '.-', 'Color', c1, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 1');
    plot(xBigSeq_T{2}(1,:), xBigSeq_T{2}(2,:), '.-', 'Color', c2, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 2');
    plot(xBigSeq_T{3}(1,:), xBigSeq_T{3}(2,:), '.-', 'Color', c3, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 3');
    plot(xBigSeq_T{4}(1,:), xBigSeq_T{4}(2,:), '.-', 'Color', c4, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 4');
    plot(xBigSeq_T{5}(1,:), xBigSeq_T{5}(2,:), '.-', 'Color', c5, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 5');
    plot(xBigSeq_T{6}(1,:), xBigSeq_T{6}(2,:), '.-', 'Color', c6, 'LineWidth', lw, 'MarkerSize', ms, 'DisplayName', 'Start 6');
    
    
    
    
    title('Top view of the function and sequence paths, problem 16, Truncated (n=2)');

    xlabel('x_1');
    ylabel('x_2');
    legend('show');
    xlim([-2.5,1]);
    ylim([-2.5, 0.5]);


    
end
    
    
    
    
    
     
    

