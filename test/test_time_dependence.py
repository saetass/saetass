import numpy as np

from saetass import Grid, Particle, Solver, State
from saetass import units as su


class TestTimeDependence:
    def test_dynamic_advection_equivalence(self):
        """Test that a callable advection velocity gives the same result as a static array."""
        num_r, r_end, t_final = 500, 10.0, 1.0
        v_const = 5.0
        r_initial_peak = 2.0
        sigma = 0.5
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 500) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)

        f_init = np.exp(-((r_raw - r_initial_peak) ** 2) / (2 * sigma**2)) * su.PSI_P

        grid = Grid(r_centers=r_grid, t_grid=t_grid)

        # 1. Static solver
        state_static = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)
        solver_static = Solver(
            state=state_static,
            problem_type="advection",
            operator_params={
                "advection": {
                    "v_centers": np.full(num_r, v_const) * su.VELOCITY,
                    "order": 1,
                    "limiter": "minmod",
                    "cfl": 0.8,
                    "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
                }
            },
            substeps={"advection": 1},
            splitting_scheme="strang",
        )
        solver_static.step(len(t_grid) - 1)

        # 2. Dynamic solver (callable)
        def v_callable(t):
            return np.full(num_r, v_const) * su.VELOCITY

        state_dynamic = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)
        solver_dynamic = Solver(
            state=state_dynamic,
            problem_type="advection",
            operator_params={
                "advection": {
                    "v_centers": v_callable,
                    "order": 1,
                    "limiter": "minmod",
                    "cfl": 0.8,
                    "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
                }
            },
            substeps={"advection": 1},
            splitting_scheme="strang",
        )
        solver_dynamic.step(len(t_grid) - 1)

        assert np.allclose(
            solver_static.state.psi_p.to_value(su.PSI_P),
            solver_dynamic.state.psi_p.to_value(su.PSI_P),
            atol=1e-12,
        )

    def test_dynamic_diffusion_equivalence(self):
        """Test that a callable diffusion coeff gives the same result as a static array."""
        num_r, r_end, t_final = 200, 10.0, 0.5
        D_const = 1.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 500) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)

        f_init = np.exp(-((r_raw - 5.0) ** 2) / 0.5) * su.PSI_P

        grid = Grid(r_centers=r_grid, t_grid=t_grid)

        # 1. Static solver
        state_static = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)
        solver_static = Solver(
            state=state_static,
            problem_type="diffusion",
            operator_params={
                "diffusion": {
                    "D_values": np.full(num_r, D_const) * su.DIFFUSION_COEFFICIENT,
                    "psi_end": 0.0 * su.PSI_P,
                }
            },
            substeps={"diffusion": 1},
            splitting_scheme="strang",
        )
        solver_static.step(len(t_grid) - 1)

        # 2. Dynamic solver
        def D_callable(t):
            return np.full(num_r, D_const) * su.DIFFUSION_COEFFICIENT

        state_dynamic = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)
        solver_dynamic = Solver(
            state=state_dynamic,
            problem_type="diffusion",
            operator_params={
                "diffusion": {
                    "D_values": D_callable,
                    "psi_end": 0.0 * su.PSI_P,
                }
            },
            substeps={"diffusion": 1},
            splitting_scheme="strang",
        )
        solver_dynamic.step(len(t_grid) - 1)

        assert np.allclose(
            solver_static.state.psi_p.to_value(su.PSI_P),
            solver_dynamic.state.psi_p.to_value(su.PSI_P),
            atol=1e-12,
        )

    def test_dynamic_source_equivalence(self):
        """Test that a callable source gives the same result as a static array."""
        num_r, r_end, t_final = 100, 10.0, 2.0
        Q_const = 2.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 100) * su.TIME

        f_init = np.zeros(num_r) * su.PSI_P

        grid = Grid(r_centers=r_grid, t_grid=t_grid)

        # 1. Static solver
        state_static = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)
        solver_static = Solver(
            state=state_static,
            problem_type="source",
            operator_params={
                "source": {
                    "source": np.full(num_r, Q_const) * su.SOURCE_PSI_P,
                }
            },
            substeps={"source": 1},
            splitting_scheme="strang",
        )
        solver_static.step(len(t_grid) - 1)

        # 2. Dynamic solver (callable taking 3 args r, p, t)
        def Q_callable(r, p, t):
            r_arr = r.value if hasattr(r, "value") else r
            return np.full_like(r_arr, Q_const) * su.SOURCE_PSI_P

        state_dynamic = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)
        solver_dynamic = Solver(
            state=state_dynamic,
            problem_type="source",
            operator_params={
                "source": {
                    "source": Q_callable,
                }
            },
            substeps={"source": 1},
            splitting_scheme="strang",
        )
        solver_dynamic.step(len(t_grid) - 1)

        assert np.allclose(
            solver_static.state.psi_p.to_value(su.PSI_P),
            solver_dynamic.state.psi_p.to_value(su.PSI_P),
            atol=1e-12,
        )
        # Analytically: f(t=2) = f(0) + 2*2 = 4
        assert np.allclose(
            solver_static.state.psi_p.to_value(su.PSI_P), 4.0, atol=1e-12
        )

    def test_genuine_time_dependent_source(self):
        """Test a genuinely time-varying source Q(t) = t."""
        num_r, r_end, t_final = 100, 10.0, 2.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 200) * su.TIME
        t_raw = t_grid.to_value(su.TIME)

        f_init = np.zeros(num_r) * su.PSI_P
        grid = Grid(r_centers=r_grid, t_grid=t_grid)

        def Q_callable(r, p, t):
            t_val = t.to_value(su.TIME) if hasattr(t, "to_value") else t
            r_arr = r.value if hasattr(r, "value") else r
            return np.full_like(r_arr, t_val) * su.SOURCE_PSI_P

        state_dynamic = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)
        solver_dynamic = Solver(
            state=state_dynamic,
            problem_type="source",
            operator_params={
                "source": {
                    "source": Q_callable,
                }
            },
            substeps={"source": 1},
            splitting_scheme="strang",
        )
        solver_dynamic.step(len(t_grid) - 1)

        # Analytical: int_0^2 t dt = 2.0. Because SourceSolver uses an explicit left Riemann sum:
        expected_sum = sum(
            t_raw[i] * (t_raw[i + 1] - t_raw[i]) for i in range(len(t_raw) - 1)
        )
        assert np.allclose(
            solver_dynamic.state.psi_p.to_value(su.PSI_P), expected_sum, atol=1e-5
        )

    def test_analytical_time_dependent_advection(self, plot_results):
        """Verifies that a time-dependent advection velocity u_w(t) = cos(t) matches the analytical solution."""
        num_r = 1000
        r_end = 10.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        r_raw = r_grid.to_value(su.LENGTH)

        t_checkpoints = [0.0, np.pi / 2, np.pi]
        t_final = np.pi

        # Ensure our time grid hits the checkpoints exactly
        t_grid_raw = np.linspace(0.0, t_final, 2000)
        t_grid_raw = np.unique(np.sort(np.append(t_grid_raw, t_checkpoints)))
        t_grid = t_grid_raw * su.TIME

        r_c = 5.0
        sigma = 0.5
        f_init = np.exp(-((r_raw - r_c) ** 2) / (sigma**2)) * su.PSI_P

        def u_w(t):
            t_val = t.to_value(su.TIME) if hasattr(t, "to_value") else t
            return np.full_like(r_raw, np.cos(t_val)) * su.VELOCITY

        grid = Grid(r_centers=r_grid, t_grid=t_grid)
        state = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)

        solver = Solver(
            state=state,
            problem_type="advection",
            operator_params={
                "advection": {
                    "v_centers": u_w,
                    "order": 2,  # Use higher order to reduce numerical diffusion
                    "limiter": "minmod",
                    "cfl": 0.8,
                    "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
                }
            },
            substeps={"advection": 1},
            splitting_scheme="strang",
        )

        current_step = 0
        saved_states = [f_init.to_value(su.PSI_P).copy()]

        for i in range(1, len(t_checkpoints)):
            target_t = t_checkpoints[i]
            # Find index in t_grid
            target_idx = np.where(t_grid_raw == target_t)[0][0]
            steps_to_take = target_idx - current_step
            if steps_to_take > 0:
                solver.step(steps_to_take)
                current_step = target_idx

            saved_states.append(solver.state.psi_p.to_value(su.PSI_P).copy())

        def analytical_solution(r, t):
            V = np.sin(t)
            # Avoid division by zero at r=0
            r_safe = r.copy()
            r_safe[r == 0] = 1e-10

            factor = ((r - V) ** 2) / (r_safe**2)
            f0 = np.exp(-((r - V - r_c) ** 2) / (sigma**2))
            ans = factor * f0
            ans[r == 0] = 0.0
            return ans

        if plot_results:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))

        for i, t in enumerate(t_checkpoints):
            num_f = saved_states[i].flatten()
            ana_f = analytical_solution(r_raw, t)

            if plot_results:
                import matplotlib.pyplot as plt

                if i == 0:
                    plt.plot(r_raw, num_f, "k--", label=f"t={t:.2f} (Num/Ana)")
                else:
                    p = plt.plot(r_raw, num_f, label=f"t={t:.2f} Numerical")
                    plt.plot(
                        r_raw,
                        ana_f,
                        "--",
                        color=p[0].get_color(),
                        label=f"t={t:.2f} Analytical",
                    )

            if i > 0:
                assert np.allclose(num_f, ana_f, atol=0.08), f"Failed at t={t}"

        if plot_results:
            import matplotlib.pyplot as plt

            plt.title("Time-Dependent Advection: u_w(t) = cos(t)")
            plt.xlabel("r")
            plt.ylabel("f(r,t)")
            plt.legend()
            plt.grid(True)
            plt.show()

    def test_analytical_time_dependent_diffusion(self, plot_results):
        """Verifies that a time-dependent diffusion coeff D(t) = D0*(1+sin(t)) matches the spherical analytical solution."""
        num_r = 1000
        r_end = 2.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        r_raw = r_grid.to_value(su.LENGTH)

        t_checkpoints = [0.0, 2.0, 5.0, 10.0]
        t_final = 10.0

        # Ensure our time grid hits the checkpoints exactly
        t_grid_raw = np.linspace(0.0, t_final, 5000)
        t_grid_raw = np.unique(np.sort(np.append(t_grid_raw, t_checkpoints)))
        t_grid = t_grid_raw * su.TIME

        D0 = 0.01
        k = np.pi

        def analytical_solution(r, t):
            tau = D0 * (t + 1.0 - np.cos(t))
            ans = np.zeros_like(r)

            # Avoid division by zero
            mask = r > 0
            ans[mask] = (np.sin(k * r[mask]) / r[mask]) * np.exp(-(k**2) * tau)

            # Limit r->0 is k * exp(-k^2 * tau)
            ans[~mask] = k * np.exp(-(k**2) * tau)
            return ans

        f_init = analytical_solution(r_raw, 0.0) * su.PSI_P

        def D_callable(t):
            t_val = t.to_value(su.TIME) if hasattr(t, "to_value") else t
            return (
                np.full_like(r_raw, D0 * (1.0 + np.sin(t_val)))
                * su.DIFFUSION_COEFFICIENT
            )

        grid = Grid(r_centers=r_grid, t_grid=t_grid)
        state = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)

        solver = Solver(
            state=state,
            problem_type="diffusion",
            operator_params={
                "diffusion": {
                    "D_values": D_callable,
                    "psi_end": 0.0 * su.PSI_P,
                }
            },
            substeps={"diffusion": 1},
            splitting_scheme="strang",
        )

        current_step = 0
        saved_states = [f_init.to_value(su.PSI_P).copy()]

        for i in range(1, len(t_checkpoints)):
            target_t = t_checkpoints[i]
            target_idx = np.where(t_grid_raw == target_t)[0][0]
            steps_to_take = target_idx - current_step
            if steps_to_take > 0:
                solver.step(steps_to_take)
                current_step = target_idx

            saved_states.append(solver.state.psi_p.to_value(su.PSI_P).copy())

        if plot_results:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))

        for i, t in enumerate(t_checkpoints):
            num_f = saved_states[i].flatten()
            ana_f = analytical_solution(r_raw, t)

            if plot_results:
                import matplotlib.pyplot as plt

                if i == 0:
                    plt.plot(r_raw, num_f, "k--", label=f"t={t:.2f} (Num/Ana)")
                else:
                    p = plt.plot(r_raw, num_f, label=f"t={t:.2f} Numerical")
                    plt.plot(
                        r_raw,
                        ana_f,
                        "--",
                        color=p[0].get_color(),
                        label=f"t={t:.2f} Analytical",
                    )

            if i > 0:
                assert np.allclose(num_f, ana_f, atol=1e-3), f"Failed at t={t}"

        if plot_results:
            import matplotlib.pyplot as plt

            plt.title("Time-Dependent Diffusion: D(t) = D0*(1+sin(t))")
            plt.xlabel("r")
            plt.ylabel("f(r,t)")
            plt.legend()
            plt.grid(True)
            plt.show()

    def test_manufactured_advection_source(self, plot_results):
        num_r = 1000
        r_end = 10.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        r_raw = r_grid.to_value(su.LENGTH)

        t_checkpoints = [0.0, np.pi, 2 * np.pi]
        t_final = 2 * np.pi

        t_grid_raw = np.linspace(0.0, t_final, 5000)
        t_grid_raw = np.unique(np.sort(np.append(t_grid_raw, t_checkpoints)))
        t_grid = t_grid_raw * su.TIME

        def f_exact(r, t):
            return (2.0 + np.cos(t)) * np.exp(-r)

        f_init = f_exact(r_raw, 0.0) * su.PSI_P

        grid = Grid(r_centers=r_grid, t_grid=t_grid)

        def u_w(t):
            t_val = t.to_value(su.TIME) if hasattr(t, "to_value") else t
            return (r_raw / (1.0 + t_val)) * su.VELOCITY

        def Q_src(r, p, t):
            t_val = t.to_value(su.TIME) if hasattr(t, "to_value") else t
            r_arr = r.to_value(su.LENGTH) if hasattr(r, "to_value") else r
            term1 = -np.sin(t_val)
            term2 = ((2.0 + np.cos(t_val)) / (1.0 + t_val)) * (3.0 - r_arr)
            return (np.exp(-r_arr) * (term1 + term2)) * su.SOURCE_PSI_P

        state = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)

        solver = Solver(
            state=state,
            problem_type="advection-source",
            operator_params={
                "advection": {
                    "v_centers": u_w,
                    "order": 2,
                    "limiter": "minmod",
                    "cfl": 0.8,
                    "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
                },
                "source": {
                    "source": Q_src,
                },
            },
            substeps={"advection": 1, "source": 1},
            splitting_scheme="strang",
        )

        current_step = 0
        saved_states = [f_init.to_value(su.PSI_P).copy()]

        for i in range(1, len(t_checkpoints)):
            target_t = t_checkpoints[i]
            target_idx = np.where(t_grid_raw == target_t)[0][0]
            steps_to_take = target_idx - current_step
            if steps_to_take > 0:
                solver.step(steps_to_take)
                current_step = target_idx

            saved_states.append(solver.state.psi_p.to_value(su.PSI_P).copy())

        if plot_results:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for i, t in enumerate(t_checkpoints):
            num_f = saved_states[i].flatten()
            ana_f = f_exact(r_raw, t)

            if plot_results:
                if i == 0:
                    ax1.plot(r_raw, num_f, "k--", label=f"t={t:.2f} (Num/Ana)")
                else:
                    p = ax1.plot(r_raw, num_f, label=f"t={t:.2f} Numerical")
                    ax1.plot(
                        r_raw,
                        ana_f,
                        "--",
                        color=p[0].get_color(),
                        label=f"t={t:.2f} Analytical",
                    )

                err = np.abs(num_f - ana_f)
                err = np.where(err < 1e-15, 1e-15, err)
                ax2.plot(r_raw, err, label=f"t={t:.2f} Abs Error")

            if i > 0:
                rel_l2 = np.sqrt(np.mean((num_f - ana_f) ** 2)) / (
                    np.sqrt(np.mean(ana_f**2)) + 1e-300
                )
                assert rel_l2 < 0.02, (
                    f"Advection manufactured test failed at t={t:.3f}: "
                    f"relative_L2={rel_l2:.4f} > 0.02"
                )

        if plot_results:
            ax1.set_title("Manufactured Advection+Source: f(r,t)")
            ax1.set_xlabel("r")
            ax1.set_ylabel("f(r,t)")
            ax1.legend()
            ax1.grid(True)

            ax2.set_title("Absolute Error |f_num - f_ana|")
            ax2.set_xlabel("r")
            ax2.set_ylabel("Error")
            ax2.set_yscale("log")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

    def test_manufactured_diffusion_source(self, plot_results):
        num_r = 1000
        r_end = 5.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        r_raw = r_grid.to_value(su.LENGTH)

        t_checkpoints = [0.0, 1.0, 3.0, 5.0]
        t_final = 5.0

        t_grid_raw = np.linspace(0.0, t_final, 5000)
        t_grid_raw = np.unique(np.sort(np.append(t_grid_raw, t_checkpoints)))
        t_grid = t_grid_raw * su.TIME

        D0 = 0.01

        def f_exact(r, t):
            return (2.0 + np.cos(t)) * np.exp(-(r**2))

        f_init = f_exact(r_raw, 0.0) * su.PSI_P

        grid = Grid(r_centers=r_grid, t_grid=t_grid)

        def D_callable(t):
            t_val = t.to_value(su.TIME) if hasattr(t, "to_value") else t
            return np.full_like(r_raw, D0 * (1.0 + t_val)) * su.DIFFUSION_COEFFICIENT

        def psi_end_callable(t):
            t_val = t.to_value(su.TIME) if hasattr(t, "to_value") else t
            return ((2.0 + np.cos(t_val)) * np.exp(-(r_end**2))) * su.PSI_P

        def Q_src(r, p, t):
            t_val = t.to_value(su.TIME) if hasattr(t, "to_value") else t
            r_arr = r.to_value(su.LENGTH) if hasattr(r, "to_value") else r
            term1 = -np.sin(t_val)
            term2 = (
                2.0
                * D0
                * (1.0 + t_val)
                * (2.0 + np.cos(t_val))
                * (3.0 - 2.0 * r_arr**2)
            )
            Q = np.exp(-(r_arr**2)) * (term1 + term2)
            return Q * su.SOURCE_PSI_P

        state = State(psi_p=f_init.copy(), grid=grid, particle=Particle.PROTON)

        solver = Solver(
            state=state,
            problem_type="diffusion-source",
            operator_params={
                "diffusion": {
                    "boundary_condition": "dirichlet",
                    "D_values": D_callable,
                    "psi_end": psi_end_callable,
                },
                "source": {
                    "source": Q_src,
                },
            },
            substeps={"diffusion": 1, "source": 1},
            splitting_scheme="strang",
        )

        current_step = 0
        saved_states = [f_init.to_value(su.PSI_P).copy()]

        for i in range(1, len(t_checkpoints)):
            target_t = t_checkpoints[i]
            target_idx = np.where(t_grid_raw == target_t)[0][0]
            steps_to_take = target_idx - current_step
            if steps_to_take > 0:
                solver.step(steps_to_take)
                current_step = target_idx

            saved_states.append(solver.state.psi_p.to_value(su.PSI_P).copy())

        if plot_results:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for i, t in enumerate(t_checkpoints):
            num_f = saved_states[i].flatten()
            ana_f = f_exact(r_raw, t)

            if plot_results:
                if i == 0:
                    ax1.plot(r_raw, num_f, "k--", label=f"t={t:.2f} (Num/Ana)")
                else:
                    p = ax1.plot(r_raw, num_f, label=f"t={t:.2f} Numerical")
                    ax1.plot(
                        r_raw,
                        ana_f,
                        "--",
                        color=p[0].get_color(),
                        label=f"t={t:.2f} Analytical",
                    )

                err = np.abs(num_f - ana_f)
                err = np.where(err < 1e-15, 1e-15, err)
                ax2.plot(r_raw, err, label=f"t={t:.2f} Abs Error")

            if i > 0:
                rel_l2 = np.sqrt(np.mean((num_f - ana_f) ** 2)) / (
                    np.sqrt(np.mean(ana_f**2)) + 1e-300
                )
                assert rel_l2 < 0.05, (
                    f"Diffusion manufactured test failed at t={t:.3f}: "
                    f"relative_L2={rel_l2:.4f} > 0.05"
                )

        if plot_results:
            ax1.set_title("Manufactured Diffusion+Source: f(r,t)")
            ax1.set_xlabel("r")
            ax1.set_ylabel("f(r,t)")
            ax1.legend()
            ax1.grid(True)

            ax2.set_title("Absolute Error |f_num - f_ana|")
            ax2.set_xlabel("r")
            ax2.set_ylabel("Error")
            ax2.set_yscale("log")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()
