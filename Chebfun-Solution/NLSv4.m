% ============================================================
% NLSE solver (Fourier spectral + RK4)
% EXPORTS NLS.mat compatible with PINN code
%
% Solves the focusing NLS:
%   i h_t + 0.5 h_xx + |h|^2 h = 0,   x in [-5, 5],  t in [0, pi/2]
%
% Exact solution: h(t,x) = 2 sech(x) exp(2it)  =>  |h| is stationary.
% Use conservation of mass and Hamiltonian to validate.
% ============================================================
clear; clc;

% ----------------------------
% Domain
% ----------------------------
L  = 10;                                    % full domain length
N  = 256;                                   % number of Fourier modes (must be even)
dx = L / N;                                 % spatial step
x  = linspace(-L/2, L/2 - dx, N).';        % (N,1) column, right endpoint omitted

% ----------------------------
% Wavenumbers  (standard 2*pi/L convention)
%
% Physical wavenumber: k_j = (2*pi/L) * j,  j in MATLAB fft order.
% Spectral Laplacian:  d^2/dx^2  <-->  multiplication by -k^2.
% ----------------------------
k  = (2*pi/L) * [0:N/2-1,  -N/2:-1].';     % (N,1)  MATLAB fft ordering
k2 = k.^2;                                 % (N,1)  reused in RHS

% ----------------------------
% Time
% ----------------------------
t0 = 0;
tf = pi/2;
dt = (pi/2) * 1e-6;                         % dt ~ 1.571e-6  (matches paper)
Nt = round((tf - t0) / dt);                % total RK4 steps  =  1 000 000

fprintf('N  = %d\n',   N);
fprintf('Nt = %d\n',   Nt);
fprintf('dt = %.4e\n', dt);

% ----------------------------
% Storage strategy
%
% save_every = floor(1e6 / 1000) = 1000
% Nt_save    = floor(1e6 / 1000) = 1000
%
% Snapshots are saved AFTER the RK4 update at step n, so
%   tt(idx) = t0 + n * dt,   n = save_every, 2*save_every, ...
% The first snapshot is at t = save_every * dt  (not t = 0).
% The last  snapshot is at t = tf              (exactly, by construction).
% ----------------------------
Nt_save_target = 1000;
save_every     = floor(Nt / Nt_save_target);   % = 1000
Nt_save        = floor(Nt / save_every);        % = 1000  (actual count)

uu = zeros(N, Nt_save);                    % (N, Nt_save)  space x time
tt = zeros(Nt_save, 1);                    % (Nt_save, 1)  snapshot times

fprintf('save_every = %d  =>  up to %d snapshots\n\n', save_every, Nt_save);

% ----------------------------
% Initial condition
% ----------------------------
h     = 2 * sech(x);                       % h(0,x) = 2 sech(x),  (N,1) real
h_hat = fft(h);                            % initial Fourier coefficients

% Validate initial condition against known max value
fprintf('IC check: max|h(0,x)| = %.10f  (exact = 2.0)\n\n', max(abs(h)));

% ----------------------------
% RHS in Fourier space
%
% NLS in physical space:
%   i h_t + 0.5 h_xx + |h|^2 h = 0
%
% Isolating d/dt in Fourier space:
%   d/dt h_hat_k = -(i/2) k^2 h_hat_k  +  i * fft(|h|^2 h)_k
%                  \____linear____/         \____nonlinear____/
%
% SOLUTION: nested anonymous function so ifft(hh) is computed ONCE per
% RHS call.  feval(@(h_phys) EXPR, ifft(hh)) binds h_phys = ifft(hh)
% and passes it into the inner lambda, where it is used twice (for |h|^2
% and for h) without a second IFFT call.
%
% Cost per RK4 stage: 1 IFFT + 1 FFT  (was: 2 IFFTs + 1 FFT).
% Total IFFTs saved over 1e6 steps: 4 * 1e6 = 4 million.
% ----------------------------
rhs = @(hh) feval( ...
    @(h_phys) 1i * ( -0.5 * k2 .* hh ...
                   + fft(abs(h_phys).^2 .* h_phys) ), ...
    ifft(hh) );

% ----------------------------
% Time integration  (explicit RK4)
%
%   k1 = F(u)
%   k2 = F(u + dt/2 * k1)
%   k3 = F(u + dt/2 * k2)
%   k4 = F(u + dt   * k3)
%   u  <- u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
%
% BUG FIX: snapshots are now saved AFTER the RK4 update with
%   tt(idx) = t0 + n * dt
% Previously they were saved before the update with (n-1)*dt, causing
% the first snapshot to sit at t = (save_every-1)*dt instead of
% t = save_every*dt, and the time vector to be shifted by one step.
% ----------------------------
idx = 1;
tic;

for n = 1 : Nt

    % --- RK4 stages ---
    k1    = rhs(h_hat);
    k2_rk = rhs(h_hat + 0.5*dt*k1);
    k3    = rhs(h_hat + 0.5*dt*k2_rk);
    k4    = rhs(h_hat + dt*k3);

    h_hat = h_hat + (dt/6) * (k1 + 2*k2_rk + 2*k3 + k4);

    % --- snapshot AFTER update: state at time t = n*dt ---
    if mod(n, save_every) == 0 && idx <= Nt_save
        uu(:, idx) = ifft(h_hat);           % physical-space field  (N,1)
        tt(idx)    = t0 + n * dt;           % exact time of this state
        idx        = idx + 1;
    end

    % --- progress report ---
    if mod(n, 100000) == 0
        fprintf('Step %7d / %d  (t = %.5f)  elapsed = %.0f s\n', ...
                n, Nt, t0 + n*dt, toc);
    end

end

elapsed = toc;
fprintf('\nDone in %.1f s (%.2f min)\n', elapsed, elapsed/60);

% --- trim trailing zeros in case floor rounding left empty columns ---
uu = uu(:, 1:idx-1);
tt = tt(1:idx-1);

% ----------------------------
% Conservation diagnostics
%
% Mass      M  = (1/N) * sum_k |h_hat_k|^2   (Parseval)
% The exact solution has constant M and H, so drift measures solver error.
% ----------------------------
M_init = sum(abs(fft(h)).^2)    / N;
M_fin  = sum(abs(h_hat).^2)     / N;
fprintf('Mass error        : %.4e  (relative)\n', abs(M_fin - M_init)/M_init);

% ----------------------------
% Save  (Python / scipy.io.loadmat compatible, v6 format)
%
%  x   (N,1)        spatial grid          — real
%  tt  (Nt_save,1)  snapshot times        — real
%  uu  (N,Nt_save)  complex field         — complex128
% ----------------------------
save('NLS1000.mat', 'x', 'tt', 'uu');

% ----------------------------
% Sanity check
% ----------------------------
disp('Saved variables:')
whos x tt uu

fprintf('\nTime range : [%.6f,  %.6f]\n', tt(1), tt(end));
fprintf('|h| range  : [%.6f,  %.6f]  (should stay approx [0, 2])\n', ...
        min(abs(uu(:))), max(abs(uu(:))));