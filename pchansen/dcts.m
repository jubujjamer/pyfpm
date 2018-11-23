function y = dcts(x)
    [n, m] = size(x);
    omega = exp(-i*pi/(2*n));
    d = [1/sqrt(2); omega.^(1:n-1).']/sqrt(2*n);
    d = d(:,ones(1,m));
    xt = [x; flipud(x)];
    yt = fft(xt);
    y = real(d.*yt(1:n, : ));
