function [xest,B,MSE] = wienerFilt(x,y,N)

X = 1/N .* fft(x(1:N));
Y = 1/N .* fft(y(1:N));
X = X(:);
Y = Y(:);
Rxx = N .* real(ifft(X .* conj(X))); % Autocorrelation function Rxx
Rxy = N .* real(ifft(X .* conj(Y))); % Crosscorrelation function Rxy
Rxx = toeplitz(Rxx); %[Rxx[0] Rxx[1]...;]
Rxy = Rxy';
B = Rxy / Rxx; B = B(:); % Wiener-Hopf eq. B = inv(Rxx) Rxy
xest = fftfilt(B,x);
xest = xest(N+1:end); % cut first N samples due to distorsion during filtering operation
MSE = mean(y(N+1:end) - xest) .^2; % mean squared error

end