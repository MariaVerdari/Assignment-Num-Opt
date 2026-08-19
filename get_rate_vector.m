function p_vec = get_rate_vector(err_seq)
% Function for the computation of the experimental rate of convergence
    len = length(err_seq);
    p_vec = NaN(len, 1); 
    
    for k = 3:len
        num = log((err_seq(k) + eps) / (err_seq(k-1) + eps));
        den = log((err_seq(k-1) + eps) / (err_seq(k-2) + eps));
        p_vec(k) = num / den;
    end
end