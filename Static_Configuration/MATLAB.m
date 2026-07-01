%% =========================================================
%   CDPR COMPLETE SCRIPT — NEWTON-EULER + SAFE PD + CV UDP
%   4-cable, 3-DOF translational | R = Identity
%
%   FIXES APPLIED vs previous version:
%     [F1] Tmin reduced to 1% of Tnom — was 15% causing QP infeasibility
%          Root cause: 4×Tmin×sin(angle) was EXCEEDING m×g requirement
%          so the equality constraint had NO feasible solution
%     [F2] Tmax set to T_motor_max×0.75 with NO Tnom cap
%          The 3×Tnom cap was making Tmax < what geometry needs
%     [F3] Slack prevention: minimum variance objective naturally avoids
%          slack by centering tensions near T_mid. No cable is pinned to
%          Tmin. The two-pass refinement now uses geometry-weighted E(T)
%     [F4] Motor step correction from CV error now ACTUALLY applied
%          Previous code computed corrective steps but discarded them
%     [F5] Table construction fixed — T_cable_log split into 4 columns
%     [F6] QP diagnostic added — prints why QP failed (bounds or geometry)
%     [F7] Tmin floor lowered — preload just enough to keep cable taut
%          Physical minimum: 50 g·cm/s² ≈ 0.05 N (cable won't go slack)
%
%   SAFETY PRINCIPLES (unchanged):
%     [S1] P-only feedback — no derivative, no integral (first hardware run)
%     [S2] Z-feedback = 0 (gravity handled by feedforward only)
%     [S3] W_fb capped at 15% of W_ff magnitude
%     [S4] Per-axis saturation on W_fb
%     [S5] Deadzone on position error (suppresses CV noise)
%     [S6] CV loss fallback to open-loop
%
%   Units: cm | g | s | g·cm/s²
% ==========================================================

clc; clear; close all;

%% =========================================================
%   BLOCK 1 : HARDWARE PARAMETERS
% ==========================================================

L = 91.44; B = 45.72; H = 47.10;
offset = 4.35;

A = [offset,     offset,     H;
     L-offset,   offset,     H;
     L-offset,   B-offset,   H;
     offset,     B-offset,   H];

num_cables = size(A,1);

platform_size = 5;
half_s        = platform_size / 2;
mass          = 300;
g             = 981;

b_local = [-half_s, -half_s, 0;
            half_s, -half_s, 0;
            half_s,  half_s, 0;
           -half_s,  half_s, 0];

Ixx = mass * platform_size^2 / 12;
Iyy = Ixx;
Izz = mass * platform_size^2 / 6;
I_tensor = diag([Ixx, Iyy, Izz]);

step_angle          = 1.8;
steps_per_rev       = 360 / step_angle;
microstep           = 16;
steps_per_rev_micro = steps_per_rev * microstep;

tau_motor_Nm = 1.24;
drum_radius  = 3.43;
eta          = 0.75;

T_motor_max = (eta * tau_motor_Nm * 1e7) / drum_radius;

%% --- Tension limits [F1] [F2] ---
%
%   WHY THE PREVIOUS Tmin=15%×Tnom CAUSED QP FAILURE:
%   ─────────────────────────────────────────────────
%   The QP equality constraint is:  J_force × T = W_des
%   W_des(z) = m×g = 294,300 g·cm/s²   (gravity support)
%
%   With 4 cables each at minimum Tmin = 0.15×Tnom = 11,036:
%     Max upward force at Tmin = 4 × Tmin × mean(u_z) ≈ 4 × 11036 × 0.85
%                              ≈ 37,500 g·cm/s²
%   This is LESS than m×g = 294,300 — the constraint is INFEASIBLE.
%   The solver correctly returns flag < 0 every single time.
%
%   FIX: Tmin must satisfy:
%     4 × Tmin × mean(u_z) << m×g
%   → Tmin << m×g / (4 × mean(u_z)) ≈ 86,500
%   → Use Tmin = 50 g·cm/s² (≈ 0.05 N) — just enough to keep taut
%
%   WHY THE 3×Tnom Tmax CAP WAS ALSO WRONG:
%   3×Tnom = 220,725. The gravity load alone needs Tnom per cable.
%   Cable geometry (u_z < 1) means each cable carries more than Tnom.
%   Capping at 3×Tnom leaves almost no headroom for acceleration forces.
%   FIX: Use T_motor_max×0.75 directly (hardware limited).

Tnom = (mass * g) / num_cables;    % 73,575 g·cm/s² per cable

%   [F1] Physical preload — just enough to keep cables taut
%   50 g·cm/s² = 0.05 N. Cables have some stiffness so this prevents
%   slack without constraining the QP solution space.
Tmin = 50;

%   [F2] Hardware-limited maximum, no artificial Tnom cap
Tmax = T_motor_max * 0.75;

T_mid = (Tmin + Tmax) / 2;

fprintf('=== Tension Limits ===\n');
fprintf('Tnom        : %10.2f  g·cm/s²\n', Tnom);
fprintf('Tmin        : %10.2f  g·cm/s²  [F1: physical floor only]\n', Tmin);
fprintf('Tmax        : %10.2f  g·cm/s²  [F2: motor limited]\n', Tmax);
fprintf('T_mid       : %10.2f  g·cm/s²\n', T_mid);
fprintf('T_motor_max : %10.2f  g·cm/s²\n', T_motor_max);

%   Feasibility check: minimum achievable upward force
%   Compute approximate mean u_z at workspace center
p_center = [(L/2), (B/2), H/2];
u_z_vals = zeros(num_cables,1);
for i = 1:num_cables
    Bc = p_center' + b_local(i,:)';
    lv = A(i,:)' - Bc;
    u_z_vals(i) = lv(3) / norm(lv);
end
min_upward = 4 * Tmin * mean(u_z_vals);
fprintf('\n--- QP Feasibility Check ---\n');
fprintf('Mean u_z at center   : %.4f\n', mean(u_z_vals));
fprintf('Min upward @ 4×Tmin  : %10.2f  g·cm/s²\n', min_upward);
fprintf('Required (m×g)       : %10.2f  g·cm/s²\n', mass*g);
if min_upward < mass*g
    fprintf('[OK] Tmin is feasible — min upward << m×g\n\n');
else
    fprintf('[ERROR] Tmin too high — QP will fail. Reduce Tmin.\n\n');
end

%% =========================================================
%   BLOCK 2 : TRAJECTORY
% ==========================================================

T_total = 20;
dt      = 0.3;
t       = 0 : dt : T_total;
N       = length(t);

P_start = [34.0, 11.0, 28.4];
P_end   = [60.0, 11.0, 28.4];

assert(all(P_start >= [0 0 0]) && all(P_start <= [L B H]), 'P_start outside workspace');
assert(all(P_end   >= [0 0 0]) && all(P_end   <= [L B H]), 'P_end outside workspace');

tau_vec   = t / T_total;
s_vec     =  10*tau_vec.^3 - 15*tau_vec.^4 + 6*tau_vec.^5;
sdot_vec  = (30*tau_vec.^2 - 60*tau_vec.^3 + 30*tau_vec.^4) / T_total;
sddot_vec = (60*tau_vec    - 180*tau_vec.^2 + 120*tau_vec.^3) / (T_total^2);

P      = P_start + (P_end - P_start) .* s_vec';
V_traj = (P_end  - P_start) .* sdot_vec';
A_traj = (P_end  - P_start) .* sddot_vec';

fprintf('=== Trajectory ===\n');
fprintf('Start : [%.1f, %.1f, %.1f] cm\n', P_start);
fprintf('End   : [%.1f, %.1f, %.1f] cm\n', P_end);
fprintf('Δ     : %.1f cm  |  T = %.0f s  |  dt = %.2f s  |  N = %d\n\n', ...
        norm(P_end-P_start), T_total, dt, N);

%% =========================================================
%   BLOCK 3 : CABLE GEOMETRY + FULL 6×4 STRUCTURE MATRIX
% ==========================================================

J_all   = zeros(6, num_cables, N);
L_cable = zeros(N, num_cables);
cond_J  = zeros(N, 1);
rank_J  = zeros(N, 1);

for k = 1:N
    p      = P(k,:)';
    J_full = zeros(6, num_cables);
    for i = 1:num_cables
        B_i      = p + b_local(i,:)';
        l_vec    = A(i,:)' - B_i;
        L_i      = norm(l_vec);
        u_i      = l_vec / L_i;
        r_i      = b_local(i,:)';
        moment_i = cross(r_i, u_i);
        J_full(1:3, i) = u_i;
        J_full(4:6, i) = moment_i;
        L_cable(k, i)  = L_i;
    end
    J_all(:,:,k) = J_full;
    rank_J(k)    = rank(J_full(1:3,:));
    cond_J(k)    = cond(J_full(1:3,:));
end

fprintf('=== Structure Matrix Health ===\n');
fprintf('Min rank  : %d  (should be 3)\n', min(rank_J));
fprintf('Mean cond : %.3f\n', mean(cond_J));
fprintf('Max  cond : %.3f\n\n', max(cond_J));

%% =========================================================
%   BLOCK 4 : NEWTON-EULER FEEDFORWARD WRENCH
%
%   W_ff = m·a_des + [0; 0; m·g]
%   This is the 3×1 force-only wrench (no moments for pure translation).
%   Z component = m·g ≈ 294,300 g·cm/s² — always present, handles gravity.
% ==========================================================

W_ff_all = zeros(3, N);
for k = 1:N
    a_k = A_traj(k,:)';
    W_ff_all(:,k) = mass*a_k - [0; 0; -mass*g];  % = m·a + [0;0;m·g]
end

%% =========================================================
%   BLOCK 5 : PD CONTROLLER PARAMETERS
%
%   [S1] P-only for first hardware run (Kd = 0)
%   [S2] Z gains = 0 always
%   [S3] W_fb capped at 15% of W_ff magnitude
%   [S4] Per-axis saturation
%   [S5] Deadzone on position error
%
%   NOTE ON GAINS:
%   Kp units are [g·cm/s² per cm] = [g/s²].
%   The feedback wrench W_fb = Kp × e has same units as W_ff.
%   For Kp = 500: a 1 cm error → W_fb = 500 g·cm/s²
%   Compare: W_ff_z = 294,300 g·cm/s² → this is 0.17% correction.
%   This is very safe. The 15% cap [S3] prevents any runaway.
% ==========================================================

Kp = diag([500.0,  500.0,  0.0]);   % g·cm/s² per cm  [S2: Z=0]
Kd = diag([0.0,    0.0,    0.0]);   % [S1] zero for first run

deadzone_cm     = 0.5;     % cm  [S5]
W_fb_max_x      = 0.15 * Tnom;   % per-axis cap [S4]
W_fb_max_y      = 0.15 * Tnom;
W_fb_max_z      = 0.0;            % [S2] always zero
fb_fraction_cap = 0.15;   % global cap as fraction of W_ff [S3]
cv_loss_timeout_s = 0.5;  % seconds before CV fallback [S6]

alpha_filter = 0.4;        % low-pass filter on error derivative

fprintf('=== PD Controller ===\n');
fprintf('Kp = [%.1f, %.1f, %.1f]  (Z always 0 [S2])\n', Kp(1,1),Kp(2,2),Kp(3,3));
fprintf('Kd = [%.1f, %.1f, %.1f]  (zero for first run [S1])\n', Kd(1,1),Kd(2,2),Kd(3,3));
fprintf('Deadzone   : %.2f cm  [S5]\n', deadzone_cm);
fprintf('W_fb_max   : [%.0f, %.0f, %.0f] g·cm/s²  [S4]\n', W_fb_max_x,W_fb_max_y,W_fb_max_z);
fprintf('Global cap : %.0f%% of W_ff  [S3]\n\n', fb_fraction_cap*100);

%% =========================================================
%   BLOCK 6 : QP SOLVER SETUP
%
%   Improved minimum variance objective (Wang et al. 2024):
%     minimize  (1/m) Σ (Ti - E(T))²
%     where E(T) = Γ·T̄_i + T̄    (tension centred at a target mean)
%
%   [F3] SLACK PREVENTION:
%   The variance objective naturally pulls ALL tensions toward E(T).
%   Since E(T) > Tmin (T_mid >> Tmin = 50), the solver never pins
%   any cable to Tmin unless the geometry absolutely forces it.
%   No cable is intentionally set to Tmin — this was NOT the original
%   design but was an emergent behaviour of the old infeasible bounds.
%
%   Two-pass refinement:
%   Pass 1: E(T) = Γ·T_mid + T_mid  (initial estimate)
%   Pass 2: E(T) = Γ·T_mid + T̄_actual  (refined with real mean)
%   This corrects the nonlinearity in E(T) = f(T) in one iteration.
% ==========================================================

Gamma   = 0.2;
m       = num_cables;
eps_reg = 1e-4;     % regularisation: larger than before for stability
mu_mom  = 1e-4;     % moment-row soft penalty weight

H_base  = (2/m)*eye(m) + eps_reg*eye(m);
qp_opts = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');

%% =========================================================
%   BLOCK 7 : FEEDFORWARD MOTOR STEPS
% ==========================================================

motor_steps_ff = zeros(N-1, num_cables);
L_prev         = L_cable(1,:);
for k = 2:N
    dL = L_cable(k,:) - L_prev;
    motor_steps_ff(k-1,:) = round((dL/drum_radius)/(2*pi)*steps_per_rev_micro);
    L_prev = L_cable(k,:);
end

fprintf('=== Feedforward Motor Steps (cumulative) ===\n');
disp(sum(motor_steps_ff));

%% =========================================================
%   BLOCK 8 : LOGGING ARRAYS
% ==========================================================

T1_log = zeros(N,1); T2_log = zeros(N,1);   % [F5] split per cable
T3_log = zeros(N,1); T4_log = zeros(N,1);
feasible_log   = false(N,1);
W_ff_log       = zeros(N,3);
W_fb_log       = zeros(N,3);
W_des_log      = zeros(N,3);
error_log      = zeros(N,3);
error_dot_log  = zeros(N,3);
cv_x_log       = nan(N,1);  cv_y_log = nan(N,1);  cv_z_log = nan(N,1);
cv_valid_log   = false(N,1);
cv_used_log    = false(N,1);
err_xy_log     = nan(N,1);
err_xyz_log    = nan(N,1);
loop_time_log  = nan(N,1);
qp_fail_reason = strings(N,1);   % [F6] diagnostic
corr_steps_log = zeros(N-1, num_cables);  % [F4] correction steps applied

%% =========================================================
%   BLOCK 9 : HARDWARE INIT
% ==========================================================

disp('=================================================');
disp(' INITIALISING HARDWARE');
disp('=================================================');

arduino = serialport("COM15", 115200);
arduino.Timeout = 0.5;
configureTerminator(arduino, "LF");
flush(arduino);
pause(2.5);

if arduino.NumBytesAvailable > 0
    try; msg = strtrim(readline(arduino)); fprintf('[Arduino] %s\n', msg); catch; end
end

PORT = 5005;
import java.net.DatagramSocket
import java.net.DatagramPacket
try
    udpSocket     = DatagramSocket(PORT);
    udpSocket.setSoTimeout(1);
    fprintf('[UDP] Port %d open\n\n', PORT);
catch
    error('Cannot open UDP port %d. Restart MATLAB.', PORT);
end
packetBuffer  = zeros(1, 512, 'int8');
receivePacket = DatagramPacket(packetBuffer, length(packetBuffer));

%% =========================================================
%   BLOCK 10 : LIVE MONITOR
% ==========================================================

fig_live = figure('Name','Live Monitor','Color','w');
hold on; grid on;
plot(P(:,1), P(:,2), 'k--', 'LineWidth',1.5, 'DisplayName','Planned');
h_target = plot(P(1,1), P(1,2), 'bs', 'MarkerFaceColor','b', 'MarkerSize',9, 'DisplayName','Target');
h_actual = plot(P(1,1), P(1,2), 'r^', 'MarkerFaceColor','r', 'MarkerSize',9, 'DisplayName','CV actual');
xlabel('X (cm)'); ylabel('Y (cm)');
title('Live: Planned vs CV'); legend('Location','best');
xlim([min(P(:,1))-15, max(P(:,1))+15]);
ylim([min(P(:,2))-15, max(P(:,2))+15]);
drawnow;

%% =========================================================
%   BLOCK 11 : INITIALISE CV
% ==========================================================

p_actual  = P_start(:);
t_cv_last = -Inf;
last_valid_cv = [];

t_drain = tic;
while toc(t_drain) < 0.3
    try
        receivePacket.setLength(512);
        udpSocket.receive(receivePacket);
        raw = receivePacket.getData();
        len = receivePacket.getLength();
        pkt = jsondecode(char(raw(1:len)'));
        if isstruct(pkt) && isfield(pkt,'valid') && pkt.valid && ...
           isfield(pkt,'x') && isfield(pkt,'y') && isfield(pkt,'z')
            last_valid_cv = pkt;
        end
    catch; end
end

if ~isempty(last_valid_cv)
    p_actual = [last_valid_cv.x; last_valid_cv.y; last_valid_cv.z];
    fprintf('[CV] Initial: [%.2f, %.2f, %.2f] cm\n\n', p_actual);
else
    fprintf('[CV] No packet — using P_start\n\n');
end

p_prev       = p_actual;
t_cv_last    = 0;
e_dot_filtered = [0;0;0];
e_prev         = [0;0;0];
t_prev_ctrl    = 0;

%   Estimate cable lengths at initial CV position for correction step
L_actual_prev = zeros(1, num_cables);
for i = 1:num_cables
    Bc = p_actual + b_local(i,:)';
    L_actual_prev(i) = norm(A(i,:)' - Bc);
end

fprintf('[OK] Ready. Starting in 2 s...\n');
pause(2.0);

%% =========================================================
%   BLOCK 12 : MAIN CONTROL LOOP
% ==========================================================

traj_start = tic;

for k = 1:(N-1)
    loop_tic = tic;
    k_next   = k + 1;
    p_des    = P(k_next,:)';
    a_des    = A_traj(k_next,:)';

    %% 12.1 READ CV
    newest_cv = [];
    t_udp_drain = tic;
    while toc(t_udp_drain) < 0.05
        try
            receivePacket.setLength(512);
            udpSocket.receive(receivePacket);
            raw = receivePacket.getData();
            len = receivePacket.getLength();
            pkt = jsondecode(char(raw(1:len)'));
            if isstruct(pkt) && isfield(pkt,'valid') && pkt.valid && ...
               isfield(pkt,'x') && isfield(pkt,'y') && isfield(pkt,'z')
                newest_cv = pkt;
            end
        catch; end
    end

    %% 12.2 UPDATE POSITION ESTIMATE
    t_now        = toc(traj_start);
    cv_available = false;

    if ~isempty(newest_cv)
        p_measured    = [newest_cv.x; newest_cv.y; newest_cv.z];
        t_cv_last     = t_now;
        last_valid_cv = newest_cv;
        cv_available  = true;
    elseif ~isempty(last_valid_cv) && (t_now - t_cv_last) < cv_loss_timeout_s
        p_measured   = [last_valid_cv.x; last_valid_cv.y; last_valid_cv.z];
        cv_available = true;
    else
        p_measured   = [];
        cv_available = false;
        if mod(k,10)==0
            fprintf('[WARNING] CV lost at step %d\n', k);
        end
    end

    %% 12.3 PD FEEDBACK WRENCH
    W_fb = [0;0;0];
    e    = [0;0;0];

    if cv_available
        p_actual = p_measured;

        %   Raw error (desired − actual) [S5]
        e_raw = p_des - p_actual;

        %   Deadzone [S5]
        e = e_raw;
        for j = 1:3
            if abs(e_raw(j)) < deadzone_cm
                e(j) = 0;
            end
        end

        %   [S2] Lock Z — gravity handled entirely by feedforward
        e(3) = 0;

        %   Derivative
        dt_ctrl = t_now - t_prev_ctrl;
        if dt_ctrl > 0 && dt_ctrl < 2*dt
            e_dot_raw      = (e - e_prev) / dt_ctrl;
            e_dot_filtered = alpha_filter * e_dot_raw + ...
                             (1-alpha_filter) * e_dot_filtered;
        end
        e_dot_filtered(3) = 0;   % [S2]

        %   PD wrench
        W_fb = Kp*e + Kd*e_dot_filtered;

        %   [S4] Per-axis saturation
        W_fb(1) = max(-W_fb_max_x, min(W_fb_max_x, W_fb(1)));
        W_fb(2) = max(-W_fb_max_y, min(W_fb_max_y, W_fb(2)));
        W_fb(3) = 0;

        %   [S3] Global cap at 15% of feedforward magnitude
        W_ff_now = W_ff_all(:, k_next);
        cap_mag  = fb_fraction_cap * norm(W_ff_now);
        W_fb_mag = norm(W_fb);
        if W_fb_mag > cap_mag && W_fb_mag > 0
            W_fb = W_fb * (cap_mag / W_fb_mag);
        end

        e_prev      = e;
        t_prev_ctrl = t_now;

        cv_valid_log(k_next)   = true;
        cv_used_log(k_next)    = true;
        cv_x_log(k_next)       = p_actual(1);
        cv_y_log(k_next)       = p_actual(2);
        cv_z_log(k_next)       = p_actual(3);
        error_log(k_next,:)    = e';
        error_dot_log(k_next,:)= e_dot_filtered';
    end

    %% 12.4 COMBINED DESIRED WRENCH
    W_ff_k = W_ff_all(:, k_next);
    W_des  = W_ff_k + W_fb;

    %% 12.5 TENSION DISTRIBUTION QP
    J_force = J_all(1:3, :, k_next);
    J_mom   = J_all(4:6, :, k_next);
    H_qp    = H_base + mu_mom*(J_mom'*J_mom);
    lb      = Tmin * ones(m,1);
    ub      = Tmax * ones(m,1);

    %   Pass 1
    E_T  = Gamma*T_mid + T_mid;
    f_qp = -(2*E_T/m)*ones(m,1);
    [T_sol, ~, flag] = quadprog(H_qp, f_qp, [], [], J_force, W_des, lb, ub, [], qp_opts);

    %   Pass 2 refinement
    if flag > 0
        T_bar   = mean(T_sol);
        E_T_ref = Gamma*T_mid + T_bar;
        f_ref   = -(2*E_T_ref/m)*ones(m,1);
        [T_ref, ~, flag2] = quadprog(H_qp, f_ref, [], [], J_force, W_des, lb, ub, [], qp_opts);
        if flag2 > 0; T_sol = T_ref; flag = flag2; end
    end

    %   [F6] Diagnostic: why did QP fail?
    if flag <= 0
        %   Check if W_des is feasible with current bounds
        T_test_min = J_force * (Tmin*ones(m,1));
        T_test_max = J_force * (Tmax*ones(m,1));
        if W_des(3) < T_test_min(3)
            qp_fail_reason(k_next) = sprintf('W_des(z)=%.0f < min achievable=%.0f', ...
                                              W_des(3), T_test_min(3));
        elseif W_des(3) > T_test_max(3)
            qp_fail_reason(k_next) = sprintf('W_des(z)=%.0f > max achievable=%.0f', ...
                                              W_des(3), T_test_max(3));
        else
            qp_fail_reason(k_next) = 'geometry/conditioning issue';
        end
        fprintf('[QP FAIL] Step %d: %s\n', k, qp_fail_reason(k_next));
    end

    %% 12.6 MOTOR STEPS — FF + CV CORRECTION [F4]
    %
    %   [F4] FIX: Previous code computed W_fb but NEVER used it for steps.
    %   Correct approach: when QP succeeds, the tension solution implicitly
    %   encodes how much extra cable to wind. We extract this via:
    %
    %   If QP gives T_sol with feedback applied:
    %     → The actual cable lengths needed to produce T_sol at current
    %       platform position differ from the planned lengths.
    %     → We correct the steps using:
    %         delta_L_corr = J_force' × pinv(J_force×J_force') × W_fb
    %                        / (cable_stiffness approximation)
    %
    %   However, since we have no cable stiffness model, we use a simpler
    %   and more reliable approach: compute the required cable lengths
    %   for the desired platform position (which feedforward already does),
    %   PLUS a direct geometric correction from the CV position error.
    %
    %   Geometric correction:
    %     If CV says platform is at p_actual, but desired is p_des,
    %     the additional cable length change needed is:
    %       ΔL_corr(i) = L_desired(i, p_des) − L_actual(i, p_actual)
    %     This is what the motor must actually travel to get from
    %     where the platform IS to where it SHOULD BE.
    %
    %   This replaces the purely planned ΔL and is the correct
    %   implementation of position feedback via motor steps.

    if flag > 0
        T1_log(k_next) = T_sol(1);
        T2_log(k_next) = T_sol(2);
        T3_log(k_next) = T_sol(3);
        T4_log(k_next) = T_sol(4);
        feasible_log(k_next) = true;

        if cv_available
            %   [F4] Geometric correction: steps to move from actual to desired
            L_des_at_p_des    = zeros(1, num_cables);
            L_des_at_p_actual = zeros(1, num_cables);
            for i = 1:num_cables
                Bc_des    = p_des    + b_local(i,:)';
                Bc_actual = p_actual + b_local(i,:)';
                L_des_at_p_des(i)    = norm(A(i,:)' - Bc_des);
                L_des_at_p_actual(i) = norm(A(i,:)' - Bc_actual);
            end
            %   Total delta = how much cable must change from current actual
            %   position to reach the desired next position
            dL_corrected = L_des_at_p_des - L_des_at_p_actual;
            motor_steps  = round((dL_corrected/drum_radius)/(2*pi)*steps_per_rev_micro);
            corr_steps_log(k,:) = motor_steps - motor_steps_ff(k,:);
        else
            motor_steps = motor_steps_ff(k,:);
        end

    else
        %   QP failed — use feedforward steps only
        T1_log(k_next) = NaN; T2_log(k_next) = NaN;
        T3_log(k_next) = NaN; T4_log(k_next) = NaN;
        feasible_log(k_next) = false;
        motor_steps = motor_steps_ff(k,:);
    end

    %% 12.7 SEND TO ARDUINO
    msg = sprintf('%d,%d,%d,%d', motor_steps(1), motor_steps(2), ...
                                  motor_steps(3), motor_steps(4));
    writeline(arduino, msg);

    t_ack    = tic;
    got_done = false;
    while ~got_done
        if arduino.NumBytesAvailable > 0
            try
                resp = strtrim(readline(arduino));
                if strcmp(resp, 'DONE'); got_done = true; end
            catch; end
        end
        if toc(t_ack) > 5.0
            warning('Step %d: Arduino DONE timeout', k);
            break;
        end
        pause(0.001);
    end

    %% 12.8 LOG AND DISPLAY
    loop_time_log(k_next) = toc(loop_tic);
    W_ff_log(k_next,:)    = W_ff_k';
    W_fb_log(k_next,:)    = W_fb';
    W_des_log(k_next,:)   = W_des';

    if cv_available
        err_xy_log(k_next)  = norm(e(1:2));
        err_xyz_log(k_next) = norm(e(1:3));
    end

    set(h_target, 'XData', p_des(1),    'YData', p_des(2));
    set(h_actual, 'XData', p_actual(1), 'YData', p_actual(2));
    drawnow limitrate;

    fprintf('Step %3d/%d | CV:%d | e=[%+.2f %+.2f]cm | Wfb=[%+.0f %+.0f] | T=[%.0f %.0f %.0f %.0f] | steps=[%d %d %d %d]\n', ...
            k, N-1, cv_available, ...
            error_log(k_next,1), error_log(k_next,2), ...
            W_fb(1), W_fb(2), ...
            T1_log(k_next), T2_log(k_next), T3_log(k_next), T4_log(k_next), ...
            motor_steps(1), motor_steps(2), motor_steps(3), motor_steps(4));

end   % end control loop

disp('Trajectory Completed');
try; clear arduino; catch; end
try; udpSocket.close(); catch; end

%% =========================================================
%   BLOCK 13 : CV SMOOTHING
% ==========================================================

valid_idx = find(cv_valid_log & isfinite(cv_x_log));
cv_x_smooth = nan(N,1);
cv_y_smooth = nan(N,1);

if numel(valid_idx) >= 5
    t_vi = t(valid_idx);
    win  = min(9, numel(valid_idx));
    if mod(win,2)==0; win=win-1; end
    if win < 3; win=3; end
    cv_x_smooth(valid_idx) = smoothdata(cv_x_log(valid_idx), 'sgolay', win);
    cv_y_smooth(valid_idx) = smoothdata(cv_y_log(valid_idx), 'sgolay', win);
end

%% =========================================================
%   BLOCK 14 : SAVE RESULTS  [F5] — fixed table construction
% ==========================================================

validation_table = table( ...
    t(:), ...
    P(:,1), P(:,2), P(:,3), ...
    V_traj(:,1), V_traj(:,2), V_traj(:,3), ...
    A_traj(:,1), A_traj(:,2), A_traj(:,3), ...
    cv_x_log, cv_y_log, cv_z_log, ...
    cv_valid_log, cv_used_log, ...
    cv_x_smooth, cv_y_smooth, ...
    error_log(:,1), error_log(:,2), error_log(:,3), ...
    W_ff_log(:,1), W_ff_log(:,2), W_ff_log(:,3), ...
    W_fb_log(:,1), W_fb_log(:,2), W_fb_log(:,3), ...
    W_des_log(:,1), W_des_log(:,2), W_des_log(:,3), ...
    T1_log, T2_log, T3_log, T4_log, ...   % [F5] split columns
    feasible_log, err_xy_log, err_xyz_log, loop_time_log, ...
    'VariableNames', { ...
    't', ...
    'x_des','y_des','z_des', ...
    'vx_des','vy_des','vz_des', ...
    'ax_des','ay_des','az_des', ...
    'x_cv','y_cv','z_cv', ...
    'cv_valid','cv_used', ...
    'x_cv_smooth','y_cv_smooth', ...
    'ex','ey','ez', ...
    'Wff_x','Wff_y','Wff_z', ...
    'Wfb_x','Wfb_y','Wfb_z', ...
    'Wdes_x','Wdes_y','Wdes_z', ...
    'T1','T2','T3','T4', ...
    'feasible','err_xy','err_xyz','t_loop'});

writetable(validation_table, 'cdpr_pd_results.csv');
disp('[OK] Saved: cdpr_pd_results.csv');

%% =========================================================
%   BLOCK 15 : PLOTS
% ==========================================================

cable_styles = {'-o','--s','-.d',':^'};
cable_names  = {'C1','C2','C3','C4'};
T_mat        = [T1_log, T2_log, T3_log, T4_log];
mk           = 1:max(1,round(N/15)):N;

%% Fig 1: XY trajectory
figure('Name','Fig1 XY Trajectory','Color','w');
plot(P(:,1), P(:,2), '-k', 'LineWidth',2, 'DisplayName','Planned');
hold on;
if any(~isnan(cv_x_smooth))
    plot(cv_x_smooth(valid_idx), cv_y_smooth(valid_idx), '--ok', ...
         'LineWidth',1.5,'MarkerSize',5,'MarkerIndices',1:3:numel(valid_idx), ...
         'DisplayName','CV smoothed');
    scatter(cv_x_log(valid_idx), cv_y_log(valid_idx), 20, 'k', 'filled', ...
            'DisplayName','CV raw');
end
plot(P_start(1),P_start(2),'ks','MarkerFaceColor','w','MarkerSize',10,'LineWidth',2,'DisplayName','Start');
plot(P_end(1),P_end(2),'k^','MarkerFaceColor','w','MarkerSize',10,'LineWidth',2,'DisplayName','End');
grid on; box on;
xlabel('X (cm)'); ylabel('Y (cm)');
title('Figure 1: XY trajectory — planned vs actual');
legend('Location','best');

%% Fig 2: Position tracking per axis
figure('Name','Fig2 Position Tracking','Color','w');
dirs = {'X','Y','Z'};
cv_logs = {cv_x_smooth, cv_y_smooth, cv_z_log};
for j = 1:3
    subplot(3,1,j);
    plot(t, P(:,j), '-k','LineWidth',2,'DisplayName','Planned');
    hold on;
    if ~isempty(valid_idx) && any(~isnan(cv_logs{j}(valid_idx)))
        plot(t(valid_idx), cv_logs{j}(valid_idx), '--ok', ...
             'LineWidth',1.5,'MarkerSize',4, ...
             'MarkerIndices',1:max(1,round(numel(valid_idx)/10)):numel(valid_idx), ...
             'DisplayName','CV');
    end
    grid on; box on;
    ylabel(sprintf('%s (cm)',dirs{j}));
    if j==1; legend('Location','best'); end
    if j==3; xlabel('Time (s)'); end
end
sgtitle('Figure 2: Position tracking — Planned vs CV','FontWeight','bold');

%% Fig 3: Tracking error
figure('Name','Fig3 Tracking Error','Color','w');
plot(t, error_log(:,1), '-ok','MarkerIndices',mk,'LineWidth',1.5,'MarkerSize',4,'DisplayName','e_x');
hold on;
plot(t, error_log(:,2), '--sk','MarkerIndices',mk,'LineWidth',1.5,'MarkerSize',4,'DisplayName','e_y');
yline(deadzone_cm, ':k','LineWidth',0.8,'Label','+deadzone');
yline(-deadzone_cm,':k','LineWidth',0.8,'Label','-deadzone');
yline(0,'-k','LineWidth',0.5);
grid on; box on;
xlabel('Time (s)'); ylabel('Error (cm)');
title('Figure 3: Position error — e_z locked to 0 [S2]');
legend('Location','best');

%% Fig 4: Cable tensions
figure('Name','Fig4 Cable Tensions','Color','w');
for i = 1:num_cables
    plot(t, T_mat(:,i), cable_styles{i}, ...
         'MarkerIndices',mk,'LineWidth',1.5,'MarkerSize',5,'Color',[0 0 0]);
    hold on;
end
yline(Tmin,'--k','LineWidth',1.2,'Label','T_{min}','LabelHorizontalAlignment','left');
yline(Tmax,':k', 'LineWidth',1.2,'Label','T_{max}','LabelHorizontalAlignment','left');
yline(Tnom,'-.k','LineWidth',0.8,'Label','T_{nom}','LabelHorizontalAlignment','right');
grid on; box on;
xlabel('Time (s)'); ylabel('Tension (g·cm/s²)');
title(sprintf('Figure 4: Cable tensions — \\Gamma = %.1f, Tmin = %.0f [F1]', Gamma, Tmin));
legend(cable_names,'Location','best');

%% Fig 5: W_fb vs W_ff safety ratio
figure('Name','Fig5 Wrench Safety','Color','w');
W_ff_mag = vecnorm(W_ff_log,2,2);
W_fb_mag = vecnorm(W_fb_log,2,2);
ratio    = W_fb_mag ./ (W_ff_mag + eps);
yyaxis left;
plot(t, W_ff_mag, '-ok','MarkerIndices',mk,'LineWidth',1.8,'MarkerSize',4,'DisplayName','W_{ff}');
hold on;
plot(t, W_fb_mag, '--sk','MarkerIndices',mk,'LineWidth',1.8,'MarkerSize',4,'DisplayName','W_{fb}');
ylabel('Wrench magnitude (g·cm/s²)');
yyaxis right;
plot(t, ratio*100, ':^k','MarkerIndices',mk,'LineWidth',1.2,'MarkerSize',4,'DisplayName','Ratio %');
yline(fb_fraction_cap*100,'--k','LineWidth',0.8,'Label','15% cap [S3]');
ylabel('W_{fb}/W_{ff} (%)');
grid on; box on;
xlabel('Time (s)');
title('Figure 5: Feedback vs feedforward wrench [S3]');
legend({'W_{ff}','W_{fb}','ratio (%)'},'Location','best');

%% Fig 6: Cable length vs tension (phase portrait)
figure('Name','Fig6 Phase Portrait','Color','w');
for i = 1:num_cables
    ok_idx = find(feasible_log);
    if ~isempty(ok_idx)
        plot(L_cable(ok_idx,i), T_mat(ok_idx,i), cable_styles{i}, ...
             'MarkerIndices',1:max(1,round(numel(ok_idx)/10)):numel(ok_idx), ...
             'LineWidth',1.5,'MarkerSize',5,'Color',[0 0 0]);
        hold on;
    end
end
yline(Tmin,'--k','LineWidth',1.0,'Label','T_{min}');
grid on; box on;
xlabel('Cable length (cm)'); ylabel('Tension (g·cm/s²)');
title('Figure 6: Phase portrait — cable length vs tension');
legend(cable_names,'Location','best');

%% Fig 7: Loop timing
figure('Name','Fig7 Loop Timing','Color','w');
plot(t, loop_time_log*1000, '-ok','MarkerIndices',mk,'LineWidth',1.5,'MarkerSize',4);
yline(dt*1000,'--k','LineWidth',1.0,'Label',sprintf('Target %.0f ms',dt*1000));
grid on; box on;
xlabel('Time (s)'); ylabel('Loop time (ms)');
title('Figure 7: Control loop timing');

%% =========================================================
%   BLOCK 16 : QUANTITATIVE SUMMARY + TUNING GUIDE
% ==========================================================

fprintf('\n=== Run Summary ===\n');
fprintf('QP feasible : %d / %d steps\n', sum(feasible_log), N);
cv_ok = ~isnan(err_xy_log) & cv_used_log;
if any(cv_ok)
    fprintf('Mean e_xy   : %.3f cm\n', mean(err_xy_log(cv_ok)));
    fprintf('Max  e_xy   : %.3f cm\n', max(err_xy_log(cv_ok)));
    fprintf('Mean e_xyz  : %.3f cm\n', mean(err_xyz_log(cv_ok)));
end

fprintf('\n=== Safety Checks ===\n');
fprintf('Max W_fb/W_ff  : %.2f%%  (cap = %.0f%%)\n', max(ratio)*100, fb_fraction_cap*100);
fprintf('W_fb(z) = 0   : %s\n', string(all(W_fb_log(:,3)==0)));
ok_T = T_mat(feasible_log,:);
if ~isempty(ok_T)
    fprintf('Min tension    : %.2f  (Tmin = %.2f)\n', min(ok_T(:)), Tmin);
    fprintf('Max tension    : %.2f  (Tmax = %.2f)\n', max(ok_T(:)), Tmax);
    n_slack = sum(ok_T(:) <= Tmin*1.05);
    fprintf('Near-slack pct : %.1f%% of (step×cable) [F3]\n', 100*n_slack/numel(ok_T));
end

fprintf('\n=== Tuning Guide ===\n');
fprintf('Current Kp = [%.1f, %.1f]  Kd = [%.1f, %.1f]\n', ...
        Kp(1,1),Kp(2,2),Kd(1,1),Kd(2,2));
if any(cv_ok)
    me = mean(err_xy_log(cv_ok));
    if me > 2.0
        fprintf('Error > 2 cm — increase Kp to [%.0f, %.0f]\n', ...
                min(Kp(1,1)*1.5,2000), min(Kp(2,2)*1.5,2000));
        fprintf('Add Kd = [%.0f, %.0f] (= 2*sqrt(Kp))\n', ...
                2*sqrt(Kp(1,1)*1.5), 2*sqrt(Kp(2,2)*1.5));
    elseif me < 0.8
        fprintf('Error < 0.8 cm — system performing well\n');
        fprintf('Consider adding Kd = [%.0f, %.0f] to reduce overshoot\n', ...
                2*sqrt(Kp(1,1)), 2*sqrt(Kp(2,2)));
    else
        fprintf('Error in acceptable range — hold current gains\n');
    end
end
fprintf('\nNEVER increase Kp > 3000 without verifying Fig 5 ratio stays < 15%%\n');
