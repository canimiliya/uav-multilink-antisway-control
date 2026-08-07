"""Formal acados OCP for the tracking-adapted SEP-NMPC baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sep_nmpc_model import PlanarParameters
from .sep_nmpc_ocp import SEPTrackingConfig


@dataclass(frozen=True)
class AcadosBuildSpec:
    parameters: PlanarParameters
    config: SEPTrackingConfig
    code_export_directory: str


def _require_acados():
    try:
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    except ImportError as exc:  # pragma: no cover - WSL-only dependency
        raise RuntimeError("formal SEP-NMPC requires CasADi and acados_template") from exc
    return ca, AcadosModel, AcadosOcp, AcadosOcpSolver


def build_formal_ocp(spec: AcadosBuildSpec):
    """Construct and compile the actual ERK/SQP_RTI OCP used by the runtime."""

    ca, AcadosModel, AcadosOcp, AcadosOcpSolver = _require_acados()
    spec.config.validate()
    p = spec.parameters
    n_p = 7  # ax_ref, previous_applied_ax, ke, rho, epsilon, m_T, m_L*l
    x = ca.MX.sym("z", 4)
    xdot = ca.MX.sym("zdot", 4)
    u = ca.MX.sym("u", 2)  # [u_ae, s_p]
    param = ca.MX.sym("p", n_p)
    ex, ev, alpha, omega = x[0], x[1], x[2], x[3]
    u_ae, slack = u[0], u[1]
    ax_ref, previous_ax, ke, rho, epsilon, m_total, ml_length = [param[i] for i in range(n_p)]
    c = ca.cos(alpha)
    s = ca.sin(alpha)
    mass = ca.vertcat(
        ca.horzcat(m_total, ml_length * c),
        ca.horzcat(ml_length * c, (ml_length / p.l) * p.l**2),
    )
    # m_L*l and m_L*l^2 are kept explicit while m_total is a runtime parameter.
    rhs = ca.vertcat(
        m_total * ax_ref + u_ae - ke * ex + ml_length * s * omega**2,
        -ml_length * p.g * s,
    )
    accelerations = ca.solve(mass, rhs)
    f_expl = ca.vertcat(ev, accelerations[0] - ax_ref, omega, accelerations[1])
    f_impl = xdot - f_expl
    ax_expr = ax_ref + (u_ae - ke * ex) / m_total
    passivity_expr = u_ae * ev + rho * ev**2 + epsilon * u_ae**2 - slack

    model = AcadosModel()
    model.name = "sep_nmpc_adapted"
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = param
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.con_h_expr = ca.vertcat(ax_expr, ax_expr - previous_ax, alpha, passivity_expr)
    model.con_h_expr_0 = model.con_h_expr
    model.con_h_expr_e = alpha
    model.cost_y_expr = ca.vertcat(x, u)
    model.cost_y_expr_e = x

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = spec.config.shooting_nodes
    ocp.solver_options.tf = spec.config.horizon_seconds
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.code_gen_options.code_export_directory = str(Path(spec.code_export_directory).resolve())

    W = np.diag([
        spec.config.w_x, spec.config.w_v, spec.config.w_alpha,
        spec.config.w_omega, spec.config.w_u, spec.config.w_s,
    ])
    ocp.cost.W = W
    ocp.cost.W_e = np.diag([
        spec.config.w_x, spec.config.w_v, spec.config.w_alpha, spec.config.w_omega,
    ])
    ocp.cost.yref = np.zeros(6)
    ocp.cost.yref_e = np.zeros(4)
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.constraints.x0 = np.zeros(4)
    ocp.constraints.lbu = np.array([-1.0e6, 0.0])
    ocp.constraints.ubu = np.array([1.0e6, spec.config.slack_max])
    ocp.constraints.idxbu = np.array([0, 1], dtype=int)
    # h=[actual ax, first-action slew, alpha, passivity residual].  The
    # second row is bounded only at stage 0; future execution uses the
    # frozen limiter because no cross-stage actuator dynamics are introduced.
    ocp.constraints.lh = np.array([-2.0, -1.0e9, -np.deg2rad(60.0), -1.0e9])
    ocp.constraints.uh = np.array([2.0, 1.0e9, np.deg2rad(60.0), 0.0])
    ocp.constraints.lh_0 = np.array([-2.0, -0.25, -np.deg2rad(60.0), -1.0e9])
    ocp.constraints.uh_0 = np.array([2.0, 0.25, np.deg2rad(60.0), 0.0])
    ocp.constraints.lh_e = np.array([-np.deg2rad(60.0)])
    ocp.constraints.uh_e = np.array([np.deg2rad(60.0)])
    ocp.constraints.x0 = np.zeros(4)

    # The stage-0 expressions retain the first-action slew bound; the
    # generic intermediate vector leaves that row unbounded.

    parameter_values = np.array([0.0, 0.0, spec.config.k_e, spec.config.rho, spec.config.epsilon, p.m_T, p.m_L * p.l])
    ocp.parameter_values = parameter_values
    solver = AcadosOcpSolver(ocp, json_file=str(Path(spec.code_export_directory) / "acados_ocp.json"))
    return solver, model, parameter_values
