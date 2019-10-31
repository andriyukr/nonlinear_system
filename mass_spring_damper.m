%% Clear

clc
clear all
close all

%% Initialize state

kend = 10000;
x(1:kend,1:2) = 0;
t = 0.001;

%% Initialize gains

K_p = 1000;
K_d = 100;

%% Initialize constants

m = 1;      % mass of the block [kg]
d = 0.1;    % damping constant [N*s/m]
s = 10;     % spring constant [N/m]
g = 9.81;   % gravitational acceleration [m/s^2]

%% Initial state values

x1_0 = -1;   % initial displacement [m]
x2_0 = -10;   % initial velocity [m/s]

%% Desired output values

y_d = 1*cos(t * (1:kend));%1*ones(1, kend);%
ydot_d = -1*sin(t * (1:kend));%zeros(1, kend);%

%% ANN parameters

ANN = 1; % 1 - ANN is enabled, 0 - ANN is disabled (only PD controller)

w = (rand(3, 1) - 0.5) / 100;
alpha = 0.1;
gamma = 0.000001;

%% Debug variables

U = 0;
U_PD = 0;
U_ANN = 0;
Alpha = alpha;
W = w';

%% Control loop

x(1,1) = x1_0;
x(1,2) = x2_0;
y = [x1_0; x2_0];

for k=1:kend-1
    
    %% Compute errors
    
    e = y_d(k) - y(1);
    e_d = ydot_d(k) - y(2);
    
    error = [e, e_d];
    error = max(min(error, 1), -1);
    
    %% PD controller
    
    u_PD = K_p * e + K_d * e_d;
    
    %% Compute the output from ANN
    
    if(ANN ~= 0)
        input = [error, -1];
        u_ANN = input/sum(abs(input)) * w;
    else
        u_ANN = 0;
    end
    
    %% Update the parameters in ANN
    
    if(ANN ~= 0)
        w = w + input'/sum(abs(input))*alpha*sign(u_PD); % SMC2
        alpha = alpha + 2 * gamma * abs(u_PD);
    end
    
    %     W = [W; w'];
    Alpha = [Alpha, alpha];
    
    %% Control inputs
    
    u = max(min(u_PD + u_ANN, 100), -100); % bound the control input between [-100, 100]
    
    U = [U, u];
    U_PD = [U_PD, u_PD];
    U_ANN = [U_ANN, u_ANN];
    
    %% System dynamics
    
    xdot(1) = x(k,2);                                       % Xdot
    xdot(2) = -s/m * x(k,1) - d/m * x(k,2) - g + 1/m * u;   % Xdotdot
    y = x(k,:);
    
    x(k + 1,:) = x(k,:) + t * xdot;
end

%% Compute errors

error = abs(y_d - x(:,1)');
mae = mean(error)
% mse = mean(error.^2)
% rmse = sqrt(mean(error.^2))

%% Plots

figure('Name', 'Trajectory tracking', 'NumberTitle', 'off');
h = plot(0:(kend-1), y_d, 0:(kend-1), x(:, 1));
%axis([0, kend-1, -1.1, 1.1]);
set(h, 'LineWidth', 2);
legend('desired', 'actual');

figure('Name', 'Error', 'NumberTitle', 'off');
h = plot(0:(kend-1), error);
%axis([0, kend-1, 0, 0.01]);
set(h, 'LineWidth', 2);

figure('Name', 'alpha', 'NumberTitle', 'off');
h = plot(0:(kend-1), Alpha);
%axis([0, kend-1, -10, 20]);
set(h, 'LineWidth', 2);

figure('Name', 'Control signal', 'NumberTitle', 'off');
h = plot(0:(kend-1), U, 0:(kend-1), U_PD, 0:(kend-1), U_ANN);
%axis([0, kend-1, -10, 20]);
set(h, 'LineWidth', 2);
legend('u', 'PD', 'ANN');