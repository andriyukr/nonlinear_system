clear all
close all
clc

load('dnn0_inverse.mat');

global system

%% Parameters
% Time for test trajectory generation
timeStart = 0;
timeEnd = 4;
system = 3;

kP = 0.5;
kI = 9.0;
kD = -0.00015;

c = [-1 0 1];
sigma1 = [0.6 0.6 0.6];
sigma2 = [0.4 0.4 0.4];
a_type2fnn = (rand(9, 2) - 0.5) / 1000;
b_type2fnn = (rand(9, 1) - 0.5) / 1000;
alpha_type2fnn = 0.01;
sigma_type2fnn = 0.01;
nu_type2fnn = 0.01;

aU = [1.5 1.5 1.5];
aL = [0.5 0.5 0.5];
c = [-1 0 1];
d = [1 1 1];
a = (rand(9, 1) - 0.5) / 1000;
b = (rand(1) - 0.5) / 1000;

dnn0 = dnn0_inverse;
dnn = dnn0_inverse;
min_max = inverse_min_max;
alpha = 0.1;

%% Initialize

% Simulation parameters
sampleTime = 0.001;
system_initial = [0, 0];

%% Test on new trajectory and run simulation

time = timeStart : sampleTime : timeEnd;
desired = 1/3*sin(3*pi*time) - 1/2*cos(2*pi*time) + 1/2;
desired_d = pi*cos(3*pi*time) + pi*sin(2*pi*time);

switch system
    case 3 % constant disturbance
        disturbance = cos(time * 2);
    case 5 % time-varying disturbance
        disturbance = -1 * ones(size(time, 2), 1);
    case 6 % uncertanties, time-varying disturbance and new relative degree
        disturbance = cos(time * 2);
    otherwise % default
        disturbance = zeros(size(time, 2), 1);
end

switch system
    case 4 % constant disturbance
        noise = normrnd(0, 0.1, [1, size(time, 2)]);
    otherwise % default
        noise = zeros(size(time, 2), 1);
end

switch system
    case {1 2 3 4 5} % constant disturbance
        relativeDegree = 1;
    case {6} % uncertanties, time-varying disturbance and new relative degree
        relativeDegree = 2;
    otherwise % default
        relativeDegree = 1;
end

%% Preallocate variables
u.inverse = zeros(1, length(time) - 1);
x.inverse = zeros(length(time), 2);
y.inverse = zeros(1, length(time));
u.pid = zeros(1, length(time) - 1);
x.pid = zeros(length(time), 2);
y.pid = zeros(1, length(time));
u.type1fnn = zeros(1, length(time) - 1);
x.type1fnn = zeros(length(time), 2);
y.type1fnn = zeros(1, length(time));
u.type2fnn = zeros(1, length(time) - 1);
x.type2fnn = zeros(length(time), 2);
y.type2fnn = zeros(1, length(time));
u.dnn0 = zeros(1, length(time) - 1);
x.dnn0 = zeros(length(time), 2);
y.dnn0 = zeros(1, length(time));
u.dnn = zeros(1, length(time) - 1);
x.dnn = zeros(length(time), 2);
y.dnn = zeros(1, length(time));

% Inizialize variables
x.inverse(1,:) = system_initial;
x.pid(1,:) = system_initial;
x.type1fnn(1,:) = system_initial;
x.type2fnn(1,:) = system_initial;
x.dnn0(1,:) = system_initial;
x.dnn(1,:) = system_initial;

pid_errorI = 0;
pid_y_old = 0;
pid_type1fnn_errorI = 0;
pid_type1fnn_y_old = 0;
pid_type2fnn_errorI = 0;
pid_type2fnn_y_old = 0;
dnn_I = 0;
dnn_y_old = 0;

%% Compare the networks

% for i = 1 : 200
%     x_1 = (i - 100) / 10;
%     x_2 = 0;
%     y_d = 1;
%     u1(i) = y_d + 0.2 * x_2 - x_1 + x_1^3;
%     u2(i) = dnn0_inverse([x_1; x_2; y_d]);
%     u3(i) = dnn0_pid([x_1; x_2; y_d]);
% end
%
% figure(4)
% clf;
% hold all;
% plot([-9.9:0.1:10], u1);
% plot([-9.9:0.1:10], u2);
% plot([-9.9:0.1:10], u3);
% % axis([-10, 10, -2, 2]);
% legend('analytical', 'DNN_0-inverse', 'DNN_0-PID');

for i = 1 : length(time) - relativeDegree
    
    y_d = desired(i + relativeDegree); % desired output
    
    %% Exact invers of the system
    switch system
        case {1 3 4 5} % nominal case, constant disturbance, time-varying disturbance
            u.inverse(i) = y_d - noise(i) + 0.2 * x.inverse(i, 2) - x.inverse(i, 1) + x.inverse(i, 1)^2 - disturbance(i);
        case 2 % uncertanties
            u.inverse(i) = 2 * (2 * y_d + 1/2 * (0.2 * x.inverse(i, 2) - x.inverse(i, 1) + x.inverse(i, 1)^2));
        case 6 % uncertanties, time-varying disturbance and new relative degree
            u.inverse(i) = 4 * (y_d - noise(i) - 0.25 * x.inverse(i, 1) + 0.25 * x.inverse(i, 1)^2 - 0.5 * disturbance(i));
        otherwise % default
            u.inverse(i) = y_d + 0.2 * x.inverse(i, 2) - x.inverse(i, 1) + x.inverse(i, 1)^2 - disturbance(i);
    end
    
    %% PID
    
    pid_error_temp = desired(i) - y.pid(i);
    pid_errorI = pid_errorI + pid_error_temp * sampleTime;
    pid_y_diff = (y.pid(i) - pid_y_old) / sampleTime;
    pid_errorD = desired_d(i) - pid_y_diff;
    pid_y_old = y.pid(i);
    
    u.pid(i) = kP * pid_error_temp + kI * pid_errorI + kD * pid_errorD;
    
    %% Type-1 FNN
    
    switch system
        case {1 3 4 5} % nominal case, constant disturbance, time-varying disturbance
            inverse_type1fnn = y_d - noise(i) + 0.2 * x.type1fnn(i, 2) - x.type1fnn(i, 1) + x.type1fnn(i, 1)^2 - disturbance(i);
        case 2 % uncertanties
            inverse_type1fnn = 2 * (2 * y_d + 1/2 * (0.2 * x.type1fnn(i, 2) - x.type1fnn(i, 1) + x.type1fnn(i, 1)^2));
        case 6 % uncertanties, time-varying disturbance and new relative degree
            inverse_type1fnn = 4 * (y_d - noise(i) - 0.25 * x.type1fnn(i, 1) + 0.25 * x.type1fnn(i, 1)^2 - 0.5 * disturbance(i));
        otherwise % default
            inverse_type1fnn = y_d + 0.2 * x.type1fnn(i, 2) - x.type1fnn(i, 1) + x.type1fnn(i, 1)^2 - disturbance(i);
    end
    
    pid_type1fnn_error_temp = desired(i) - y.type1fnn(i);
    pid_type1fnn_errorI = pid_type1fnn_errorI + pid_type1fnn_error_temp * sampleTime;
    pid_type1fnn_y_diff = (y.type1fnn(i) - pid_type1fnn_y_old) / sampleTime;
    pid_type1fnn_errorD = desired_d(i) - pid_type1fnn_y_diff;
    pid_type1fnn_y_old = y.type1fnn(i);
    
    u.pid_type1fnn(i) = kP * pid_type1fnn_error_temp + kI * pid_type1fnn_errorI + kD * pid_type1fnn_errorD;
    
    u.type1fnn(i) = 0.6 * u.pid_type1fnn(i) + 0.4 * inverse_type1fnn;
    
    %% Type-2 FNN
    
    pid_type2fnn_error_temp = desired(i) - y.type2fnn(i);
    pid_type2fnn_errorI = pid_type2fnn_errorI + pid_type2fnn_error_temp * sampleTime;
    pid_type2fnn_y_diff = (y.type2fnn(i) - pid_type2fnn_y_old) / sampleTime;
    pid_type2fnn_errorD = desired_d(i) - pid_type2fnn_y_diff;
    pid_type2fnn_y_old = y.type2fnn(i);
    
    u.pid_type2fnn(i) = kP * pid_type2fnn_error_temp + kI * pid_type2fnn_errorI + kD * pid_type2fnn_errorD;

    pid_type2fnn_errors = max(min([pid_type2fnn_error_temp pid_type2fnn_errorD], 1), -1)';
    
    % Fuzzification
       
%     mfU = exp(-(repmat(pid_type2fnn_errors, [1 3]) - c).^2./(2*sigma1.^2));
%     mfL = exp(-(repmat(pid_type2fnn_errors, [1 3]) - c).^2./(2*sigma2.^2));
    
    mu_e_1_upper = elliptic(aU(1), c(1), d(1), pid_type2fnn_errors(1));
    mu_e_1_lower = elliptic(aL(1), c(1), d(1), pid_type2fnn_errors(1));
    mu_e_2_upper = elliptic(aU(2), c(2), d(2), pid_type2fnn_errors(1));
    mu_e_2_lower = elliptic(aL(2), c(2), d(2), pid_type2fnn_errors(1));
    mu_e_3_upper = elliptic(aU(3), c(3), d(3), pid_type2fnn_errors(1));
    mu_e_3_lower = elliptic(aL(3), c(3), d(3), pid_type2fnn_errors(1));
    mu_de_1_upper = elliptic(aU(1), c(1), d(1), pid_type2fnn_errors(2));
    mu_de_1_lower = elliptic(aL(1), c(1), d(1), pid_type2fnn_errors(2));
    mu_de_2_upper = elliptic(aU(2), c(2), d(2), pid_type2fnn_errors(2));
    mu_de_2_lower = elliptic(aL(2), c(2), d(2), pid_type2fnn_errors(2));
    mu_de_3_upper = elliptic(aU(3), c(3), d(3), pid_type2fnn_errors(2));
    mu_de_3_lower = elliptic(aL(3), c(3), d(3), pid_type2fnn_errors(2));
    
    % Firing levels
    
%     wU = repmat(mfU(1,:), [1 3]) .* reshape(repmat(mfU(2,:), [3 1]), [1, 9]);    
%     wL = repmat(mfL(1,:), [1 3]) .* reshape(repmat(mfL(2,:), [3 1]), [1, 9]);
%     wUtilde = wU ./ sum(wU);
%     wLtilde = wL ./ sum(wL);
    
    wU(1) = mu_e_1_upper * mu_de_1_upper;
    wU(2) = mu_e_1_upper * mu_de_2_upper;
    wU(3) = mu_e_1_upper * mu_de_3_upper;
    wU(4) = mu_e_2_upper * mu_de_1_upper;
    wU(5) = mu_e_2_upper * mu_de_2_upper;
    wU(6) = mu_e_2_upper * mu_de_3_upper;
    wU(7) = mu_e_3_upper * mu_de_1_upper;
    wU(8) = mu_e_3_upper * mu_de_2_upper;
    wU(9) = mu_e_3_upper * mu_de_3_upper;
    wL(1) = mu_e_1_lower * mu_de_1_lower;
    wL(2) = mu_e_1_lower * mu_de_2_lower;
    wL(3) = mu_e_1_lower * mu_de_3_lower;
    wL(4) = mu_e_2_lower * mu_de_1_lower;
    wL(5) = mu_e_2_lower * mu_de_2_lower;
    wL(6) = mu_e_2_lower * mu_de_3_lower;
    wL(7) = mu_e_3_lower * mu_de_1_lower;
    wL(8) = mu_e_3_lower * mu_de_2_lower;
    wL(9) = mu_e_3_lower * mu_de_3_lower;
    
    % Compute the output
    
%     f = a_type2fnn * pid_type2fnn_errors + b_type2fnn;
%     u.type2fnn(i) = 0.5 * wUtilde * f + 0.5 * wLtilde * f;

    wU_tilde = wU / sum(wU);
    wL_tilde = wL / sum(wL);

    u.type2fnn(i) = (wU_tilde * a + wU_tilde * a) / 2;
       
    % Update the parameters
    
%     F1 = 0.5 * wLtilde + 0.5 * wUtilde;
%     F = F1 ./ F1 .* F1;
%     a_type2fnn = a_type2fnn - repmat(pid_type2fnn_errors', [9 1]) .* repmat(F' .* alpha_type2fnn .* sign(u.pid_type2fnn(i)), [1 2]) * sampleTime;
%     b_type2fnn = b_type2fnn - F' * alpha_type2fnn * sign(u.pid_type2fnn(i)) * sampleTime;
%     alpha_type2fnn = alpha_type2fnn + (4 * sigma_type2fnn * abs(u.pid_type2fnn(i)) - nu_type2fnn * sigma_type2fnn * alpha_type2fnn) * sampleTime;

    PI = (0.5 * wU_tilde + 0.5 * wL_tilde);
    da = - (0.5 * wU_tilde + 0.5 * wL_tilde) / (PI * PI') * alpha_type2fnn * sign(u.pid_type2fnn(i));
    db = - 1 / (PI * PI') * alpha_type2fnn * sign(u.pid_type2fnn(i));
    dalpha_type2fnn = 2 * sigma_type2fnn * abs(u.pid_type2fnn(i)) - nu_type2fnn * sigma_type2fnn * alpha_type2fnn;
    
    a = a + da' * sampleTime;
    b = b + db * sampleTime;
    alpha_type2fnn = alpha_type2fnn + dalpha_type2fnn * sampleTime;
    
    u.type2fnn(i) = u.pid_type2fnn(i) - u.type2fnn(i);
    
    switch system
        case {1 3 4 5} % nominal case, constant disturbance, time-varying disturbance
            inverse_type2fnn = y_d - noise(i) + 0.2 * x.type2fnn(i, 2) - x.type2fnn(i, 1) + x.type2fnn(i, 1)^2 - disturbance(i);
        case 2 % uncertanties
            inverse_type2fnn = 2 * (2 * y_d + 1/2 * (0.2 * x.type2fnn(i, 2) - x.type2fnn(i, 1) + x.type2fnn(i, 1)^2));
        case 6 % uncertanties, time-varying disturbance and new relative degree
            inverse_type2fnn = 4 * (y_d - noise(i) - 0.25 * x.type2fnn(i, 1) + 0.25 * x.type2fnn(i, 1)^2 - 0.5 * disturbance(i));
        otherwise % default
            inverse_type2fnn = y_d + 0.2 * x.type2fnn(i, 2) - x.type2fnn(i, 1) + x.type2fnn(i, 1)^2 - disturbance(i);
    end
    
    u.type2fnn(i) = 0.5 * u.pid_type2fnn(i) + 0.5 * inverse_type2fnn;
    
    %% DNN_0 alone
    
    dnn0_x = [x.dnn0(i, 1); x.dnn0(i, 2); y_d];
    u.dnn0(i) = dnn0(dnn0_x);
    
    %% DNN
    
    dnn_error_temp = desired(i) - y.dnn(i);
    dnn_y_diff = (y.dnn(i) - dnn_y_old) / sampleTime;
    dnn_errorD = desired_d(i) - dnn_y_diff;
    dnn_y_old = y.dnn(i);
    delta_u1 = alpha * dnn_error_temp;
    delta_u2 = alpha * (1/2 * dnn_error_temp + 0.1 * dnn_errorD - 0.5 * abs(dnn_error_temp) * dnn_errorD);
    
    %     u_1 = desired(i + 1) + 0.2 * x_online2(i, 2) - x_online2(i, 1) + x_online2(i, 1)^3;
    %     u_2 = sim(net_online, [desired(i + 1), x_online2(i,:)]);
    %     net_online = adapt(net_online, [desired(i + 1), x_online2(i,:)], u_2 + update21);
    %     u_2 = sim(net_online, [desired(i + 1), x_online2(i,:)]);
    
    dnn_I = dnn_I + delta_u1;
    dnn_x = [x.dnn(i, 1); x.dnn(i, 2); y_d];
    u.dnn(i) = dnn(dnn_x) + dnn_I;
    
    %% Updates the system
    
    %fprintf('%d -> error = %.4f: [%.4f, %.4f, %.4f]\n', i, error, u_1, u_2, u_online2(i));
    
    % Update the system controlled by the system inverse
    [x_temp_inverse, y_temp_inverse] = nonlinearSystemEqn(x.inverse(i,:)', u.inverse(i), disturbance(i), noise(i));
    x.inverse(i + 1, :) = x_temp_inverse';
    y.inverse(i + 1) = y_temp_inverse;
    
    % Update the system controlled by PID
    [x_temp_pid, y_temp_pid] = nonlinearSystemEqn(x.pid(i,:)', u.pid(i), disturbance(i), noise(i));
    x.pid(i + 1, :) = x_temp_pid';
    y.pid(i + 1) = y_temp_pid;
    
    % Update the system controlled by FNN Gaussian
    [x_temp_type1fnn, y_temp_type1fnn] = nonlinearSystemEqn(x.type1fnn(i,:)', u.type1fnn(i), disturbance(i), noise(i));
    x.type1fnn(i + 1, :) = x_temp_type1fnn';
    y.type1fnn(i + 1) = y_temp_type1fnn;
    
    % Update the system controlled by FNN Gaussian
    [x_temp_type2fnn, y_temp_type2fnn] = nonlinearSystemEqn(x.type2fnn(i,:)', u.type2fnn(i), disturbance(i), noise(i));
    x.type2fnn(i + 1, :) = x_temp_type2fnn';
    y.type2fnn(i + 1) = y_temp_type2fnn;
    
    % Update the system controlled by DNN0
    [x_temp_dnn0, y_temp_dnn0] = nonlinearSystemEqn(x.dnn0(i,:)', u.dnn0(i), disturbance(i), noise(i));
    x.dnn0(i + 1, :) = x_temp_dnn0';
    y.dnn0(i + 1) = y_temp_dnn0;
    
    % Update the system controlled by DNN
    [x_temp_dnn, y_temp_dnn] = nonlinearSystemEqn(x.dnn(i,:)', u.dnn(i), disturbance(i), noise(i));
    x.dnn(i + 1, :) = x_temp_dnn';
    y.dnn(i + 1) = y_temp_dnn;
    
    fprintf('completed: %.4f%% \n', (100 * i / (length(time) - relativeDegree)));
    
end

%% Compute error

error.inverse = sqrt(mean((y.inverse - desired).^2)); % with analytical
error.pid = sqrt(mean((y.pid - desired).^2)); % without PID
error.type1fnn = sqrt(mean((y.type1fnn - desired).^2)); % without FNN Gaussian
error.type2fnn = sqrt(mean((y.type2fnn - desired).^2)); % without FNN Elliptic
error.dnn0 = sqrt(mean((y.dnn0 - desired).^2)); % with DNN0
error.dnn = sqrt(mean((y.dnn - desired).^2)); % with DNN

error

%% Plots
figure(1)
clf;
grid on;
hold on;

colors = lines(7);

h2 = plot(time, y.inverse, 'color', colors(5,:), 'linewidth', 2);
h3 = plot(time, y.pid, 'color', colors(2,:), 'linewidth', 2);
h4 = plot(time, y.type1fnn, 'color', colors(4,:), 'linewidth', 2);
h5 = plot(time, y.type2fnn, 'color', colors(7,:), 'linewidth', 2);
h6 = plot(time, y.dnn0, 'color', colors(3,:), 'linewidth', 2);
h7 = plot(time, y.dnn, 'color', colors(1,:), 'linewidth', 2);
h1 = plot(time, desired, 'color', [0.25, 0.25, 0.25], 'LineStyle', '--', 'linewidth', 2);
h8 = plot(0, 0, 'color', [1, 1, 1], 'Visible', 'off');

% legend([h1, h2, h3, h4, h8, h5, h6, h7], '$y^*$', strcat('Eq. (', num2str(3 + system * 2),')'), 'PID', 'T1-FNN', '', 'IT2-FNN', 'DFNN$_0$', 'DFNN', 'Orientation', 'horizontal', 'Location', 'northoutside', 'FontSize', 15, 'Interpreter', 'latex', 'NumColumns', 4);
legend([h1, h2, h3, h4, h8, h5, h6, h7], '$y^*$', 'inverse', 'PID', 'T1-FNN', '', 'IT2-FNN', 'DFNN$_0$', 'DFNN', 'Orientation', 'horizontal', 'Location', 'northoutside', 'FontSize', 15, 'Interpreter', 'latex', 'NumColumns', 4);
xlabel('Time [s]', 'fontsize', 15, 'Interpreter', 'latex');
ylabel('System''s output', 'fontsize', 15, 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex')
set(gca, 'fontsize', 15);
switch system
    case 1 % constant disturbance
        ylim([-0.2 1.4]);
    case 2 % time-varying disturbance
        ylim([-0.2 1.4]);
    case 3 % uncertanties, time-varying disturbance and new relative degree
        ylim([-0.5 1.5]);
    case 4 % default
        ylim([-0.5 1.6]);
    case 5 % default
        ylim([-1.1 1.4]);
end
print(strcat('results/nonlinear/simulation_response_case', num2str(system), '.eps'), '-depsc', '-r300');
print(strcat('results/nonlinear/simulation_response_case', num2str(system), '.png'), '-dpng', '-r300');

%% Plots
figure(2)
clf;
grid on;
hold on;

h1 = plot(time(1:end - 1), u.inverse, 'color', colors(5,:), 'linewidth', 2);
h2 = plot(time(1:end - 1), u.pid, 'color', colors(2,:), 'linewidth', 2);
h3 = plot(time(1:end - 1), u.type1fnn, 'color', colors(4,:), 'linewidth', 2);
h4 = plot(time(1:end - 1), u.type2fnn, 'color', colors(7,:), 'linewidth', 2);
h5 = plot(time(1:end - 1), u.dnn0, 'color', colors(3,:), 'linewidth', 2);
h6 = plot(time(1:end - 1), u.dnn, 'color', colors(1,:), 'linewidth', 2);

%legend([h1, h2, h3, h4, h5, h6], strcat('Eq. (', num2str(3 + system * 2),')'), 'PID', 'T1-FNN', 'IT2-FNN', 'DFNN$_0$', 'DFNN', 'Orientation', 'horizontal', 'Location', 'northoutside', 'FontSize', 15, 'Interpreter', 'latex', 'NumColumns', 3);
legend([h1, h2, h3, h4, h5, h6], 'inverse', 'PID', 'T1-FNN', 'IT2-FNN', 'DFNN$_0$', 'DFNN', 'Orientation', 'horizontal', 'Location', 'northoutside', 'FontSize', 15, 'Interpreter', 'latex', 'NumColumns', 3);
xlabel('Time [s]', 'fontsize', 15, 'Interpreter', 'latex');
ylabel('System''s input', 'fontsize', 15, 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex')
set(gca, 'fontsize', 15);
switch system
    case 1 % constant disturbance
        ylim([-0.1 2.8]);
    case 2 % time-varying disturbance
        ylim([-0.6 6.7]);
    case 3 % uncertanties, time-varying disturbance and new relative degree
        ylim([-1.0 3.8]);
    case 4 % default
        ylim([-0.4 3.5]);
    case 5 % default
        ylim([-0.3 3.8]);
end
print(strcat('results/nonlinear/simulation_control_case', num2str(system), '.eps'), '-depsc', '-r300');
print(strcat('results/nonlinear/simulation_control_case', num2str(system), '.png'), '-dpng', '-r300');

%% Plots
figure(3)
clf;
grid on;
hold on;

h1 = plot(time, abs(y.inverse - desired), 'color', colors(5,:), 'linewidth', 2);
h2 = plot(time, abs(y.pid - desired), 'color', colors(2,:), 'linewidth', 2);
h3 = plot(time, abs(y.type1fnn - desired), 'color', colors(4,:), 'linewidth', 2);
h4 = plot(time, abs(y.type2fnn - desired), 'color', colors(7,:), 'linewidth', 2);
h5 = plot(time, abs(y.dnn0 - desired), 'color', colors(3,:), 'linewidth', 2);
h6 = plot(time, abs(y.dnn - desired), 'color', colors(1,:), 'linewidth', 2);

% legend([h1, h2, h3, h4, h5, h6], strcat('Eq. (', num2str(3 + system * 2),')'), 'PID', 'T1-FNN', 'IT2-FNN', 'DFNN$_0$', 'DFNN', 'Orientation', 'horizontal', 'Location', 'northoutside', 'FontSize', 15, 'Interpreter', 'latex', 'NumColumns', 3);
legend([h1, h2, h3, h4, h5, h6], 'inverse', 'PID', 'T1-FNN', 'IT2-FNN', 'DFNN$_0$', 'DFNN', 'Orientation', 'horizontal', 'Location', 'northoutside', 'FontSize', 15, 'Interpreter', 'latex', 'NumColumns', 3);
xlabel('Time [s]', 'fontsize', 15, 'Interpreter', 'latex');
ylabel('$|$Error$|$', 'fontsize', 15, 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex')
set(gca, 'fontsize', 15);
switch system
    case 1 % constant disturbance
        ylim([0 0.6]);
    case 2 % time-varying disturbance
        ylim([0 1.1]);
    case 3 % uncertanties, time-varying disturbance and new relative degree
        ylim([0 1.0]);
    case 4 % default
        ylim([0 0.6]);
    case 5 % default
        ylim([0 1.1]);
end
print(strcat('results/nonlinear/simulation_error_case', num2str(system), '.eps'), '-depsc', '-r300');
print(strcat('results/nonlinear/simulation_error_case', num2str(system), '.png'), '-dpng', '-r300');