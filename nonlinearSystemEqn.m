function [x, y] = nonlinearSystemEqn(x, u, d, n)

global system

switch system
    case 1 % nominal case
        x = [x(2); x(1) - x(1)^2] + [0; 1] * u;
        y = (-0.2 * x(1) + x(2));
    case 2 % uncertanties
        x = 1/2 * [x(2); x(1) - x(1)^2] + 1/2 * [0; 1] * u;
        y = 1/2 * (-0.2 * x(1) + x(2));
    case {3 5} % constant disturbance, time-varying disturbance
        x = [x(2); x(1) - x(1)^2] + [0; 1] * u + [0; 1] * d;
        y = (-0.2 * x(1) + x(2));
    case 4 % new relative degree
        x = [x(2); x(1) - x(1)^2] + [0; 1] * u;
        y = (-0.2 * x(1) + x(2)) + n;
    case 6 % uncertanties, time-varying disturbance and new relative degree
        x = 0.5 * [x(2); x(1) - x(1)^2] + 0.5*[0; 1] * u + [0; 1] * d;
        y = x(1);
    otherwise % default
        x = [x(2); x(1) - x(1)^2] + [0; 1] * u;
        y = (-0.2 * x(1) + x(2));
end

end