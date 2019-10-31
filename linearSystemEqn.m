function [x, y] = linearSystemEqn(x, u, d)

global system_A system_B system_C system_D system_E

x = system_A * x + system_B * u + system_E * d;
y = system_C * x + system_D * u;

end

