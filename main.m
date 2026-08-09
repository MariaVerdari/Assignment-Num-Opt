% PROBLEM 5, MODIFIED NEWTON

clear
clc
close all


% load("test_nonlinsys.mat")


% set random seed

rng(358655)


% function, gradient and Hessian


f = @(x) problem_5(x);
gradf = @(x) get_gradient(x);
Hessf = @(x) get_hessian(x);


function g = get_gradient(x)
    [~, g] = problem_5(x);
end

function H = get_hessian(x)
    [~, ~, H] = problem_5(x);
end


% fine tuned parameters

kmax = 5000; %bo
tolgrad =  1e-8; %bo
c1 = 1e-4;
rho = 0.8;
btmax = 50;
beta = 1e-3; % nelle note 
jmax = 2000;



dims = [2, 1e3, 1e4, 1e5];

for n = dims
    start_points = [- ones(n,1), unifrnd(-2, 0, n), unifrnd(-2, 0, n), unifrnd(-2, 0, n), unifrnd(-2, 0, n), unifrnd(-2, 0, n)]; % deignated starting point and 5 randomly generated

    for x0 = start_points
        tic
        [xk,fk,gradfk_norm,k,xseq, btseq, flag_multi] = modified_newton_bcktrck(x0,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, beta, jmax);
        time = toc;


        if k >= 3
            x_k   = xseq(:, end);
            x_k1  = xseq(:, end-1); 
            x_k2  = xseq(:, end-2); 
            x_k3  = xseq(:, end-3); 
            
            e_k   = norm(x_k - x_k1);
            e_k1  = norm(x_k1 - x_k2);
            e_k2  = norm(x_k2 - x_k3);
            
            % experimental rate
            rate_exp = log((e_k + eps) / (e_k1 + eps)) / log((e_k1 + eps) / (e_k2 + eps));
        else
            rate_exp = Nan;

        end
            

        print("Starting point:", x0, "Grad norm:",gradfk_norm, "Iters/max iters:", k, "Flag:", flag_multi, "Rate of convergence:",rate_exp ,"Time:", time)
        print()
    end

end

%%

%figures


  figure
     [X,Y] = meshgrid(linspace(-6,2,500), linspace(-3,5,500));
    
    Z1 = zeros(500,500);
    Z2 = zeros(500,500);

    for i = 1:500
        for j = 1:500
            Z1(i,j) = f1([X(1,i); Y(j,1)]);
            Z2(i,j) = f2([X(1,i); Y(j,1)]);

        end
    end
    contour(X, Y, Z1)
hold on
    contour(X, Y, Z2)
    plot(xseq1(1,:),xseq1(2, :), col = 'red' )
    plot(xseq2(1, :),xseq2(2, :), col = 'blue' )
    plot(xseq3(1,:),xseq1(2, :), col = 'red' )
    plot(xseq4(1, :),xseq2(2, :), col = 'blue' )



    figure
   
    surf(X,Y,Z1,'FaceAlpha',0.5,'EdgeColor','none') 

    hold on

    surf(X,Y,Z2,'FaceAlpha',0.5,'EdgeColor','none') 

    plot3(xseq1(1,:), xseq1(2,:), f1([xseq1(1,:); xseq1(2,:)]),col = 'red')
    plot3(xseq2(1,:), xseq2(2,:), f1([xseq2(1,:); xseq2(2,:)]),col = 'blue')


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