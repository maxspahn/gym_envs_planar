import casadi as ca
from casadi import cos, sin
import numpy as np
from sys import argv


# create variables
def main():
    #n = int(argv[1])
    n = 20
    print("Creating dynamics for pendulum with " + str(n) + " joints, point masses")
    name = '../output/dynamics_' + str(n) + "_Reacher.casadi"
    dynamics = createDynamics(n)
    dynamics.save(name)

def createDynamics(n):
    q = ca.SX.sym('q', n)
    qdot = ca.SX.sym('qdot', n)
    tau = ca.SX.sym('tau', n)
    l = ca.SX.sym('l', n)
    m = ca.SX.sym('m', n)
    k = ca.SX.sym('k', 1)
    g = ca.SX.sym('g', 1)
    x = ca.SX.sym('x', (n, 2))
    v = ca.SX.sym('v', (n, 2))
    v_2 = ca.SX.sym('v_2', n)

    # velocities and positions
    v[0, 0] = -1 * sin(q[0]) * l[0] * qdot[0]
    v[0, 1] = cos(q[0]) * l[0] * qdot[0]
    x[0, 0] = cos(q[0]) * l[0]
    x[0, 1] = sin(q[0]) * l[0]

    for i in range(1, n):
        angle = 0.0
        for j in range(i+1):
            angle += q[j]
        v[i, 0] = v[i-1, 0] - sin(angle) * l[i] * qdot[i]
        v[i, 1] = v[i-1, 1] + cos(angle) * l[i] * qdot[i]
        x[i, 0] = x[i-1, 0] + cos(angle) * l[i]
        x[i, 1] = x[i-1, 1] + sin(angle) * l[i]

    for i in range(n):
        v_2[i] = ca.norm_2(v[i, :]) ** 2



    # kinetic energy
    T = sum([0.5 * m[i] * v_2[i] for i in range(n)])

    # potential energy
    V = sum([g * m[i] * x[i, 1] for i in range(n)])

    # lagrangian
    L = T - V

    # derivatives
    dL_dq = ca.gradient(L, q)
    dL_dqdot = ca.gradient(L, qdot)

    d2L_dq2 = ca.jacobian(dL_dq, q)
    d2L_dqdqdot = ca.jacobian(dL_dq, qdot)
    d2L_dqdot2 = ca.jacobian(dL_dqdot, qdot)


    K = np.identity(n) * k
    # equation of motion 
    # M * q_ddot + F q_dot - f = tau
    # augmented for first order system
    tau_aug = ca.vertcat(np.zeros((n, 1)), tau)
    f_aug = ca.vertcat(np.zeros((n, 1)), dL_dq)
    F_aug = ca.horzcat(np.zeros((2*n, n)), ca.vertcat(-1*np.identity(n), d2L_dqdqdot))
    K_aug = ca.horzcat(np.zeros((2*n, n)), ca.vertcat(0*np.identity(n), K))
    M_aug = ca.vertcat(ca.horzcat(np.identity(n), np.zeros((n, n))), ca.horzcat(np.zeros((n, n)), d2L_dqdot2))

    x = ca.vertcat(q, qdot)

    rhs = tau_aug + f_aug - ca.mtimes(F_aug, x) - ca.mtimes(K_aug, x)
    xdot = ca.solve(M_aug, rhs)

    # save system
    dynamics = ca.Function("dynamics", [q, qdot, l, m, g, k, tau], [xdot])
    return dynamics

if __name__ == "__main__":
    main()
