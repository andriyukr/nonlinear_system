clear all
close all
clc

%% Global variables

global system_A system_B system_C system_D system_E

%% Parameters
% Time for test trajectory generation
timeStart = 0;
timeEnd = 100;
system = 1;
alpha = 1;

% Plots
lineWidth = 2;
fontSize = 10;
fontSizeLegend = 10;
markerSize = 8;
grey = [1 1 1].*200/255;
white = [255,255,255]./255;
%whitebg(white);

%% Initialize

% Load data
load('./info/model.mat');
load('./info/linearSystemParameters.mat');
load('./info/trainingData_combined.mat');

sampleTime = systemParameters.sampleTime;
system_A = systemParameters.system_A;
system_B = systemParameters.system_B;
system_C = systemParameters.system_C;
system_D = systemParameters.system_D;
system_E = systemParameters.system_E;
system_initial = systemParameters.system_initial;
timeDelay = systemParameters.timeDelay;

%% Test on new trajectory and run simulation

time = timeStart : sampleTime : timeEnd;
desired = sin(time * 2 * pi / 15) + cos(time * 2 * pi / 12) - 1;

% Modify parameters

% system_A(2,2) = -0.8;

switch system
    case 2 % uncertanties
        system_A = 1/2*system_A;
        system_B = 1/2*system_B;
        system_C = 1/2*system_C;
        relativeDegree = 1;
    case 5 % new relative degree
        system_C = [1 0];
        relativeDegree = 2;
    case 6 % uncertanties, time-varying disturbance and new relative degree
        system_A = 1/2*system_A;
        system_B = 1/2*system_B;
        system_C = [1 0];
        relativeDegree = 2;
    otherwise % default
        relativeDegree = 1;
end

switch system
    case 3 % constant disturbance
        disturbance = -2 * ones(size(time, 2), 1);
    case 4 % time-varying disturbance
        disturbance = -2 * cos(time * 2);
    case 6 % uncertanties, time-varying disturbance and new relative degree
        disturbance = -2 * cos(time * 2);
    otherwise % default
        disturbance = zeros(size(time, 2), 1);
end

eig(system_A)

dcgain = system_D + system_C * inv(ones(2) - system_A) * system_B

%% Get new reference

numInputs = net.numInputs;

% Preallocate variables
input_est = zeros(numInputs, length(time) - 1);
input_est_online = zeros(6, length(time) - 1);
u = zeros(1, length(time) - 1);
u_any = zeros(1, length(time) - 1);
u_dnn = zeros(1, length(time) - 1);
u_online1 = zeros(1, length(time) - 1);
u_online2 = zeros(1, length(time) - 1);
u_online21 = zeros(1, length(time) - 1);
u_online22 = zeros(1, length(time) - 1);
output_est = zeros(1, length(time) - 1);
output_est_analytical = zeros(1, length(time) - 1);
x = zeros(length(time), 2);
x_noDNN = zeros(length(time), 2);
x_any = zeros(length(time), 2);
x_online1 = zeros(length(time), 2);
x_online2 = zeros(length(time), 2);
y = zeros(1, length(time));
y_noDNN = zeros(1, length(time));
y_any = zeros(1, length(time));
y_online1 = zeros(1, length(time));
y_online2 = zeros(1, length(time));

% Inizialize variables
x(1,:) = system_initial;
y(1) = 0;
x_noDNN(1,:) = system_initial;
y_noDNN(1) = 0;
x_any(1,:) = system_initial;
y_any(1) = 0;
x_online1(1,:) = system_initial;
y_online1(1) = 0;
x_online2(1,:) = system_initial;
y_online2(1) = 0;
errorI = 0;
errorOld = 0;
errorOld2 = desired(1) - y_online2(1);
update1 = 0.0;
update2 = 0.0;
update22 = 0.0;

data_withNN.y(1,1) = y(1);
data_withNN.x(1,:) = x(1,:);
data_withNN.y_any(1,1) = y_any(1);
data_withNN.x_any(1,:) = x_any(1,:);
data_withNN.y_online1(1,1) = y_online1(1);
data_withNN.x_onlien1(1,:) = x_online1(1,:);
data_withNN.y_online2(1,1) = y_online2(1);
data_withNN.x_onlien2(1,:) = x_online2(1,:);

net_online = onlineNN([3 10 1]);
dnn_online = net;
gamma = 0.1;
adaptive_alpha = alpha;
Alpha = [];

cAb = system_C * system_A^(relativeDegree - 1) * system_B;
cA = system_C * system_A^(relativeDegree);
cAe = system_C * system_A^(relativeDegree - 1) * system_E;

for i = 1 : length(time) - relativeDegree
    
    %% Exact invers of the system
    y_d_temp = desired(i + relativeDegree); % desired output
    
    u_any(i) = (1 / cAb) * (-cA * x_any(i,:)' + y_d_temp - cAe * disturbance(i)); % output from inverse dynamics controller
    
    %% DNN alone
    
    input_est(:,i) = [desired(i + 1), x(i, :)]';
    u_dnn(i) = cell2mat(net(num2cell(input_est(:,i)))); % output from DNN (should approximate u)
    
    %     if i > 1
    %         for j = 1:100
    %             [net, Y] = adapt(net, [y(i), x(i - 1, :)]', u_dnn(i - 1));
    %         end
    %         fprintf('u_past = %.4f -> u_curr = %.4f\n', u_dnn(i - 1), Y);
    %         u_new = cell2mat(net(num2cell(input_est(:,i))));
    %         fprintf('u_old = %.4f -> u_new = %.4f\n', u_dnn(i), u_new);
    %     end
    
    %% Online learning 1
    
    error = desired(i) - y_online1(i);
    errorD = (error - errorOld);
    errorOld = error;
    update1 = update1 + alpha * error;
    update2 = update2 + alpha * (1/2 * error + errorD - 1/2 * abs(error) * errorD);
    
    %     u_1 = cell2mat(net(num2cell([desired(i + 1), x_online1(i,:)]')));
    
    u_online1(i) = update1;
    
    %% Online learning 2
    
    error = desired(i) - y_online2(i);
    errorD = (error - errorOld2);
    errorOld2 = error;
    update21 = adaptive_alpha * error;
    update22 = adaptive_alpha * (1/2 * error + errorD - 1/2 * abs(error) * errorD);
    
    if i > 20
        relativeDegree_estimated = 1;
        U = [x_online2(i - 1 - relativeDegree_estimated - 2: i - 1 - relativeDegree_estimated, :), u_online2(i - 1 - relativeDegree_estimated - 2: i - 1 - relativeDegree_estimated)'];
        Y = y_online2(i - 1 - 2 : i - 1)';
        F = pinv(U) * Y;
        f_estimated = F(1:2);
        g_estimated = F(3);
        f_real = cA;
        g_real = cAb;
        
        fprintf('f_estimated(1) = %.4f == f_real(1) = %.4f, f_estimated(2) = %.4f == f_real(2) = %.4f, g_estimated = %.4f == g_real = %.4f\n', f_estimated(1), f_real(1), f_estimated(2), f_real(2), g_estimated, g_real);
    end
    
    u_1 = cell2mat(dnn_online(num2cell([desired(i + 1), x_online2(i,:)]')));
    u_2 = sim(net_online, [desired(i + 1), x_online2(i,:)]);
    net_online = adapt(net_online, [desired(i + 1), x_online2(i,:)], u_2 + update21);
    u_2 = sim(net_online, [desired(i + 1), x_online2(i,:)]);
    
    %     adaptive_alpha = max(adaptive_alpha - gamma * error * sign(errorD), 0);
    Alpha = [Alpha, adaptive_alpha];
    
    %     if i == 1
    %         u_2 = 1;
    %     end
    %     if i > 1
    %         u_old = sim(net_online, [desired(i + 1), x_online2(i,:)]);
    %         net_online = adapt(net_online, [y_online2(i), x_online2(i - 1, :)], u_online22(i - 1));
    %         test22 = sim(net_online, [y_online2(i), x_online2(i - 1, :)]);
    %         fprintf('desired = %.4f -> actual = %.4f ---> difference = %.4f\n', u_online22(i - 1), test22, test22 - u_online22(i - 1));
    %         u_2 = sim(net_online, [desired(i + 1), x_online2(i,:)]);
    %         fprintf('u_old = %.4f -> u_new = %.4f ---> difference = %.4f\n', u_old, u_2, u_2 - u_old);
    %     end
    
    u_online2(i) = u_1 + u_2;
    u_online21(i) = u_1;
    u_online22(i) = u_2;
    
    %%
    
    %fprintf('%d -> error = %.4f: [%.4f, %.4f, %.4f]\n', i, error, u_1, u_2, u_online2(i));
    
    % Update the system without DNN
    [x_temp_noDNN, y_temp_noDNN] = linearSystemEqn(x_noDNN(i,:)', desired(i + 1), disturbance(i));
    x_noDNN(i + 1, :) = x_temp_noDNN';
    y_noDNN(i + 1) = y_temp_noDNN;
    
    % Update the system controlled by the system inverse
    [x_temp_any, y_temp_any] = linearSystemEqn(x_any(i,:)', u_any(i), disturbance(i));
    x_any(i + 1, :) = x_temp_any';
    y_any(i + 1) = y_temp_any;
    
    % Update the system controlled by DNN
    [x_temp, y_temp] = linearSystemEqn(x(i,:)', u_dnn(i), disturbance(i));
    x(i + 1, :) = x_temp';
    y(i + 1) = y_temp;
    
    % Update the system controlled by online DNN
    [x_temp_online, y_temp_online] = linearSystemEqn(x_online1(i,:)', u_online1(i), disturbance(i));
    x_online1(i + 1, :) = x_temp_online';
    y_online1(i + 1) = y_temp_online;
    
    % Update the system controlled by online
    [x_temp_online, y_temp_online] = linearSystemEqn(x_online2(i,:)', u_online2(i), disturbance(i));
    x_online2(i + 1, :) = x_temp_online';
    y_online2(i + 1) = y_temp_online;
end

data_withNN.time = time(1:end - 1)';
data_withNN.y = y(1:end - 1)';
data_withNN.x = x(1:end - 1,:);
data_withNN.y_any = y_any(1:end - 1)';
data_withNN.x_any = x_any(1:end - 1,:);
data_withNN.y_online1 = y_online1(1:end - 1)';
data_withNN.x_online1 = x_online1(1:end - 1,:);
data_withNN.y_online2 = y_online2(1:end - 1)';
data_withNN.x_online2 = x_online2(1:end - 1,:);
data_withNN.y_d = desired(1:end - 1)';

%% Compute error

error_withoutDNN = sqrt(mean((y_noDNN(1:end-1)' - data_withNN.y_d).^2)) % without DNN
error_withDNN = sqrt(mean((data_withNN.y - data_withNN.y_d).^2)) % with DNN
error_online1 = sqrt(mean((data_withNN.y_online1 - data_withNN.y_d).^2)) % with online DNN
error_online2 = sqrt(mean((data_withNN.y_online2 - data_withNN.y_d).^2)) % with online DNN
error_any = sqrt(mean((data_withNN.y_any - data_withNN.y_d).^2)) % with analytical

% dcgain
% cAb

%% Plots
figure(1)
clf;
hold all;

plot(data_withNN.time, data_withNN.y_d, 'k:', 'linewidth', 2);
plot(data_withNN.time, y_noDNN(1:end-1), 'color', grey, 'linewidth', 2);
plot(data_withNN.time, data_withNN.y_any, 'color', [0.4660, 0.6740, 0.1880], 'linewidth', 2);
plot(data_withNN.time, data_withNN.y, 'color', [0, 0.4470, 0.7410], 'linewidth', 2);
plot(data_withNN.time, data_withNN.y_online1, 'color', [0.4940, 0.1840, 0.5560], 'linewidth', 2);
plot(data_withNN.time, data_withNN.y_online2, 'color', [0.8500, 0.3250, 0.0980], 'linewidth', 2);

comment = sprintf('RMSE (without DNN) = %.3g\nRMSE (analytical) = %.3g\nRMSE (with DNN) = %.3g\nRMSE (online learning) = %.3g\nRMSE (DNN and online learning) = %.3g', error_withoutDNN, error_any, error_withDNN, error_online1, error_online2);
text(0.05,0.15,comment,'units','normalized','fontsize', 10, 'Interpreter', 'latex');
legend('Desired', 'Actual (without DNN)', 'Actual (analytical)', 'Actual (with DNN)', 'Actual (online learning)', 'Actual (DNN and online learning)', 'location', 'ne', 'FontSize', 10, 'Interpreter', 'latex');
xlabel('Time [s]', 'fontsize', fontSize, 'Interpreter', 'latex');
ylabel('Position', 'fontsize', fontSize, 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex')
set(gca, 'fontsize', fontSize);
print(strcat('results/linear/response_case', num2str(system), '_', num2str(timeEnd), 's.eps'), '-depsc', '-r300');
print(strcat('results/linear/response_case', num2str(system), '_', num2str(timeEnd), 's.png'), '-dpng', '-r300');

%%
figure(2)
clf;
hold all;

plot(data_withNN.time(1 : end - 1), u_any(1 : end - 1), 'color', [0.4660, 0.6740, 0.1880], 'linewidth', 2);
plot(data_withNN.time(1 : end - 1), u_dnn(1 : end - 1), 'color', [0, 0.4470, 0.7410], 'linewidth', 2);
plot(data_withNN.time(1 : end - 1), u_online1(1 : end - 1), 'color', [0.4940, 0.1840, 0.5560], 'linewidth', 2);
plot(data_withNN.time(1 : end - 1), u_online21(1 : end - 1), 'color', [0.3010, 0.7450, 0.9330], 'linestyle', ':', 'linewidth', 2);
plot(data_withNN.time(1 : end - 1), u_online22(1 : end - 1), 'color', [0.9290, 0.6940, 0.1250], 'linestyle', '--', 'linewidth', 2);
plot(data_withNN.time(1 : end - 1), u_online2(1 : end - 1), 'color', [0.8500, 0.3250, 0.0980], 'linewidth', 2);

l = legend('Actual (analytical)', 'Actual (with DNN)', 'Actual (online learning)', 'Actual ($u_1$)', 'Actual ($u_2$)', 'Actual ($u_1 + u_2$)');
set(l, 'location', 'northeast', 'FontSize', 10, 'Interpreter', 'latex');
xlabel('Time [s]', 'fontsize', fontSize, 'Interpreter', 'latex');
ylabel('Control', 'fontsize', fontSize, 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex')
set(gca, 'fontsize', fontSize);
print(strcat('results/linear/control_case', num2str(system), '_', num2str(timeEnd), 's.eps'), '-depsc', '-r300');
print(strcat('results/linear/control_case', num2str(system), '_', num2str(timeEnd), 's.png'), '-dpng', '-r300');

%%
figure(3)
clf;
hold all;

plot(data_withNN.time, abs(y_noDNN(1:end-1)' - data_withNN.y_d), 'color', grey, 'linewidth', 2);
plot(data_withNN.time, abs(data_withNN.y_any - data_withNN.y_d), 'color', [0.4660, 0.6740, 0.1880], 'linewidth', 2);
plot(data_withNN.time, abs(data_withNN.y - data_withNN.y_d), 'color', [0, 0.4470, 0.7410], 'linewidth', 2);
plot(data_withNN.time, abs(data_withNN.y_online1 - data_withNN.y_d), 'color', [0.4940, 0.1840, 0.5560], 'linewidth', 2);
plot(data_withNN.time, abs(data_withNN.y_online2 - data_withNN.y_d), 'color', [0.8500, 0.3250, 0.0980], 'linewidth', 2);

legend('Actual (without DNN)', 'Actual (analytical)', 'Actual (with DNN)', 'Actual (online learning)', 'Actual (DNN and online learning)', 'location', 'ne', 'FontSize', 10, 'Interpreter', 'latex');
xlabel('Time [s]', 'fontsize', fontSize, 'Interpreter', 'latex');
ylabel('Error', 'fontsize', fontSize, 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex')
set(gca, 'fontsize', fontSize);
print(strcat('results/linear/error_case', num2str(system), '_', num2str(timeEnd), 's.eps'), '-depsc', '-r300');
print(strcat('results/linear/error_case', num2str(system), '_', num2str(timeEnd), 's.png'), '-dpng', '-r300');

%%
figure(4)
plot(Alpha, 'linewidth', 2)