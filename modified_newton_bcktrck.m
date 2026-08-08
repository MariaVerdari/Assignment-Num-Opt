function [xk,fk,gradfk_norm,k,xseq, btseq] = modified_newton_bcktrck(x0,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax)
%Implementation of the Newton method with backtracking for optimization


k = 0;
xk =x0;
gradfk_norm = norm(gradf(xk));
btseq = zeros(kmax,1);
xseq = zeros(length(x0), kmax); %storing the sequence

while(gradfk_norm >= tolgrad && k < kmax)
    bt = 0;
    %new descent direction
    pk = pcg(Hessf(xk), - gradf(xk)); %solution of linear system with conj grad method, assuming hessian is symm and pos def 

    %check if pk is descent direction
    if (pk'*gradf(xk)>0)
        break
    end

    alpha = 1; %initialization
    xnew = xk + alpha *pk;

    %backtracking
    while (bt<=btmax && f(xnew)> f(xk)- c1*alpha*gradf(xk)'*gradf(xk) )
        bt = bt+1;
        alpha = alpha*rho;
        xnew = xk + alpha *pk; 
    end

    if (f(xnew)> f(xk)- c1*alpha*gradf(xk)'*gradf(xk)) %if the last step does not satisfy armijo I break everything (we should restart the method with different parameters)
        break
    end 

    btseq(k+1)= bt;
    xk = xnew; 
    k = k+1;
    gradfk_norm = norm(gradf(xk));
    xseq(:, k) = xk;
end
fk = f(xk);
%xseq = xseq(:, 1:k); %slice
xseq = [x0, xseq]; %concatenation


end