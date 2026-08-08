function [xk,fk,gradfk_norm,k,xseq, btseq, flag_multi] = truncated_newton_bcktrck(x0,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, jmax, etak)
%Implementation of the truncated Newton method with backtracking for optimization

% definire etak

k = 0;
xk =x0;
gradfk_norm = norm(gradf(xk));
btseq = zeros(kmax,1);
xseq = zeros(length(x0), kmax); %storing the sequence
flag_multi = 0; %everything works

while(gradfk_norm >= tolgrad && k < kmax)
    bt = 0;

    
    Bk = Hessf(xk);

    z = 0;
    
    ck = - gradf(xk);

    d = ck;

    r = ck; 

    j = 0;

    while(norm(r) > etak*norm(ck) && j <jmax)
        if d' * Bk* d > 0 % if pos def
            %pk = pcg ???
            alpha = (r' *r) / (d'* Bk *d) ;
            z = z + alpha * d;
            rnew = r - alpha * Bk * d;
            beta = (rnew' * rnew)/(r'*r);
            d = rnew + beta*d;
            r = rnew;
        else
            if j == 0
               z = -gradf(xk);                    
            end
            break
        end      
    j = j+1;
    end

    %new descent direction

    pk = z;


    %check if pk is descent direction (SHOULD ALWAYS BE)
    if (pk'*gradf(xk)>0)
        flag_multi = 1; %no descent
        break
    end

    alpha = 1; %initialization
    xnew = xk + alpha *pk;

    %backtracking
    while (bt<=btmax && f(xnew)> f(xk)+ c1*alpha*pk'*gradf(xk) )
        bt = bt+1;
        alpha = alpha*rho;
        xnew = xk + alpha *pk; 
    end

    if (f(xnew)> f(xk)+ c1*alpha*pk'*gradf(xk)) %if the last step does not satisfy armijo I break everything (we should restart the method with different parameters)
        flag_multi = 2; %no armijo
        break
    end 

    btseq(k+1)= bt;
    xk = xnew; 
    k = k+1;
    gradfk_norm = norm(gradf(xk));
    xseq(:, k) = xk;
end
fk = f(xk);
xseq = xseq(:, 1:k); %slice
xseq = [x0, xseq]; %concatenation


end