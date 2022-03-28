# pylint: disable-all
import casadi as ca
from casadi import cos, sin
import numpy as np
from sys import argv


# create variables
def main():
    # n = int(argv[1])
    n = 15
    print(
        "Creating dynamics for pendulum with "
        + str(n)
        + " joints, point masses"
    )
    name = "./" + str(n) + "_Reacher.casadi"
    dynamics, _, _ = createDynamics(n)
    dynamics.save(name)


def createDynamics(n):
    q = ca.SX.sym("q", n)
    qdot = ca.SX.sym("qdot", n)
    qddot = ca.SX.sym("qddot", n)
    tau = ca.SX.sym("tau", n)
    l = ca.SX.sym("l", n)
    m = ca.SX.sym("m", n)
    k = ca.SX.sym("k", 1)
    g = ca.SX.sym("g", 1)
    fk = ca.SX.sym("x", (n, 2))
    v = ca.SX.sym("v", (n, 2))
    v_2 = ca.SX.sym("v_2", n)

    # velocities and positions
    v[0, 0] = -1 * sin(q[0]) * l[0] * qdot[0]
    v[0, 1] = cos(q[0]) * l[0] * qdot[0]
    fk[0, 0] = cos(q[0]) * l[0]
    fk[0, 1] = sin(q[0]) * l[0]

    for i in range(1, n):
        angle = 0.0
        for j in range(i + 1):
            angle += q[j]
        v[i, 0] = v[i - 1, 0] - sin(angle) * l[i] * qdot[i]
        v[i, 1] = v[i - 1, 1] + cos(angle) * l[i] * qdot[i]
        fk[i, 0] = fk[i - 1, 0] + cos(angle) * l[i]
        fk[i, 1] = fk[i - 1, 1] + sin(angle) * l[i]

    for i in range(n):
        v_2[i] = ca.norm_2(v[i, :]) ** 2

    # kinetic energy
    T = sum([0.5 * m[i] * v_2[i] for i in range(n)])

    # potential energy
    V = sum([g * m[i] * fk[i, 1] for i in range(n)])

    # lagrangian
    L = T - V

    # derivatives
    dL_dq = ca.gradient(L, q)
    dL_dqdot = ca.gradient(L, qdot)

    d2L_dq2 = ca.jacobian(dL_dq, q)
    d2L_dqdqdot = ca.jacobian(dL_dq, qdot)
    d2L_dqdot2 = ca.jacobian(dL_dqdot, qdot)

    M = d2L_dqdot2
    F = d2L_dqdqdot
    f = dL_dq
    # damping
    K = np.identity(n) * k

    tau_forward = (
        ca.mtimes(M, qddot) + ca.mtimes(F, qdot) + ca.mtimes(K, qdot) - f
    )
    # equation of motion
    # M * q_ddot + F q_dot - f = tau
    # augmented for first order system
    tau_aug = ca.vertcat(np.zeros((n, 1)), tau)
    f_aug = ca.vertcat(np.zeros((n, 1)), f)
    F_aug = ca.horzcat(np.zeros((2 * n, n)), ca.vertcat(-1 * np.identity(n), F))
    K_aug = ca.horzcat(np.zeros((2 * n, n)), ca.vertcat(0 * np.identity(n), K))
    M_aug = ca.vertcat(
        ca.horzcat(np.identity(n), np.zeros((n, n))),
        ca.horzcat(np.zeros((n, n)), M),
    )

    x = ca.vertcat(q, qdot)

    rhs = tau_aug + f_aug - ca.mtimes(F_aug, x) - ca.mtimes(K_aug, x)
    xdot = ca.solve(M_aug, rhs)

    dynamics = ca.Function("dynamics", [q, qdot, l, m, g, k, tau], [xdot])
    fk = ca.Function("fk", [q, l], [fk])
    tau_fun = ca.Function("tau", [q, qdot, l, m, g, k, qddot], [tau_forward])
    return dynamics, fk, tau_fun


if __name__ == "__main__":
    main()
