function [xk,fk,gradfk_norm,k,xseq, btseq, flag_multi] = modified_newton_bcktrck(x0,f,gradf,Hessf,kmax,tolgrad,c1, rho, btmax, beta, jmax)
%Implementation of the modified Newton method with backtracking for optimization


%AGGIUNGERE PRINT ALPHA E TAU

k = 0;
xk =x0;
n = length(x0); %dimension
gradfk_norm = norm(gradf(xk));
btseq = zeros(kmax,1);

if (n==2)
    xseq = zeros(n, kmax); %storing all the sequence for the plot
else
    xseq = zeros(n, 4); %storing the last 4 of the sequence for memory reasons
    xseq(:,4) = x0;
end

flag_multi = 0; %everything works

while(gradfk_norm >= tolgrad && k < kmax)
    bt = 0;

    % finding Bk

    H = Hessf(xk);
    d = diag(H);
    
    tauk = 0;

    if (any(d<=0))
        tauk = beta - min(d);
    end
    
    for j = 0:jmax
        [R ,  flag] = chol(H + (tauk*speye(n)) ); %try to compute choleski decomposition
        if flag ==0 %symmetric positive definite
            break
        else
            tauk = max(2*tauk, beta);
        end



    end

    if (j == jmax && flag ~=0)
        flag_multi = 1; %not pos def
        break
    end





    %Bk = H + (tauk*speye(n));


 
    %new descent direction

    y = (R')\(-gradf(xk));
    pk =  R \ y;

    %check if pk is descent direction 
    if (pk'*gradf(xk)>0)
        flag_multi = 2; %flag if not descent direction
        break
    end

    alpha = 1; %initialization
    xnew = xk + alpha *pk;

    %backtracking
    while (bt<=btmax && f(xnew)> f(xk) + c1*alpha*pk'*gradf(xk) )
        bt = bt+1;
        alpha = alpha*rho;
        xnew = xk + alpha *pk; 
    end

    if (f(xnew)> f(xk) + c1*alpha*pk'*gradf(xk)) %if the last step does not satisfy armijo I break everything (we should restart the method with different parameters)
        flag_multi = 3; %flag if not Armijo
        break
    end 

    btseq(k+1)= bt;
    xk = xnew; 
    k = k+1;
    gradfk_norm = norm(gradf(xk));

    if n ==2
        xseq(:,k) = xk;
    else
        xseq = [xseq(:, 2:4), xnew]; %shifting
    end
end
fk = f(xk);

if n==2
    xseq = xseq(:,1:k); %slice
    xseq = [x0, xseq]; %concatenation
end


end