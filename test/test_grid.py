import numpy as np
import pytest

from saetass import units as su
from saetass.grid import Grid


class TestGrid:
    def test_missing_initialization(self):
        with pytest.raises(ValueError):
            Grid()

    def test_init_from_centers_and_faces(self):
        # Only r_centers
        r_c = np.array([1.0, 2.0, 3.0]) * su.LENGTH
        g1 = Grid(r_centers=r_c)
        assert g1.shape == (3,)
        assert g1.r_faces is not None
        assert len(g1.r_faces) == 4

        # Only p_centers
        p_c = np.array([5.0, 10.0]) * su.MOMENTUM
        g2 = Grid(p_centers=p_c, is_p_log=False)
        assert g2.shape == (2,)

        # Both
        g3 = Grid(r_centers=r_c, p_centers=p_c, is_p_log=False)
        assert g3.shape == (2, 3)

        # Null cases for properties
        assert g2.dr is None
        assert g1.dp is None
        assert g2.volumes is None
        assert g2.face_areas is None

    def test_single_cell_initialization(self):
        g = Grid(
            r_centers=[1.0] * su.LENGTH, p_centers=[2.0] * su.MOMENTUM, is_p_log=False
        )
        assert g.dr[0].to_value(su.LENGTH) == 1.0
        assert g.dp[0].to_value(su.MOMENTUM) == 1.0

    def test_negative_p_faces_approximation(self):
        # p=0 center leads to negative left face extrapolation, and should ValueError
        with pytest.raises(ValueError):
            Grid(p_centers=[0.0, 1.0] * su.MOMENTUM, is_p_log=False)

    def test_uniform_grid_validations(self):
        with pytest.raises(ValueError, match="together"):
            Grid.uniform(
                r_min=1.0 * su.LENGTH, r_max=2.0 * su.LENGTH
            )  # Missing num_cells

        with pytest.raises(ValueError, match="negative"):
            Grid.uniform(r_min=-1.0 * su.LENGTH, r_max=2.0 * su.LENGTH, num_r_cells=10)

        with pytest.raises(ValueError, match="strictly greater"):
            Grid.uniform(r_min=2.0 * su.LENGTH, r_max=1.0 * su.LENGTH, num_r_cells=10)

        with pytest.raises(ValueError, match="positive integer"):
            Grid.uniform(r_min=1.0 * su.LENGTH, r_max=2.0 * su.LENGTH, num_r_cells=0)

        # Same for p
        with pytest.raises(ValueError, match="strictly greater"):
            Grid.uniform(
                p_min=2.0 * su.MOMENTUM, p_max=1.0 * su.MOMENTUM, num_p_cells=10
            )

        # Missing both r and p
        with pytest.raises(ValueError, match="fully specified"):
            Grid.uniform(t_min=0 * su.TIME, t_max=1 * su.TIME, num_timesteps=10)

        # Wrong temp bounds
        with pytest.raises(ValueError, match="together"):
            Grid.uniform(
                r_min=1.0 * su.LENGTH,
                r_max=2.0 * su.LENGTH,
                num_r_cells=10,
                t_min=1 * su.TIME,
            )

        with pytest.raises(ValueError, match="positive integer"):
            Grid.uniform(
                r_min=1.0 * su.LENGTH,
                r_max=2.0 * su.LENGTH,
                num_r_cells=10,
                t_min=0 * su.TIME,
                t_max=1 * su.TIME,
                num_timesteps=-1,
            )

        with pytest.raises(ValueError, match="strictly greater"):
            Grid.uniform(
                r_min=1.0 * su.LENGTH,
                r_max=2.0 * su.LENGTH,
                num_r_cells=10,
                t_min=1 * su.TIME,
                t_max=0 * su.TIME,
                num_timesteps=10,
            )

    def test_log_grid(self):
        g = Grid.log_spaced(
            r_min=1.0 * su.LENGTH,
            r_max=100.0 * su.LENGTH,
            num_r_cells=3,
            p_min=50.0 * su.MOMENTUM,
            p_max=100.0 * su.MOMENTUM,
            num_p_cells=3,
        )
        assert g.is_log_p
        assert np.isclose(g.p_centers[0], np.log10(50.0))

        with pytest.raises(ValueError, match="strictly positive"):
            Grid.log_spaced(
                r_min=0.0 * su.LENGTH, r_max=10.0 * su.LENGTH, num_r_cells=10
            )

        with pytest.raises(ValueError, match="strictly positive"):
            Grid.log_spaced(
                p_min=0.0 * su.MOMENTUM, p_max=10.0 * su.MOMENTUM, num_p_cells=10
            )

    def test_non_uniform_clustering(self):
        g = Grid.non_uniform_clustering(
            r_min=0 * su.LENGTH,
            r_max=10 * su.LENGTH,
            num_r_cells=100,
            cluster_center=5.0 * su.LENGTH,
            cluster_width=1.0 * su.LENGTH,
        )
        assert g.r_faces[0] == 0.0 * su.LENGTH
        assert g.r_faces[-1] == 10.0 * su.LENGTH

    def test_properties(self):
        t_g = np.linspace(0, 1, 11) * su.TIME
        g = Grid(r_centers=[1.0, 2.0] * su.LENGTH, t_grid=t_g)
        assert g.num_timesteps == 10

        with pytest.raises(ValueError):
            g_bad = Grid(r_centers=[1.0] * su.LENGTH, t_grid=[0] * su.TIME)
            _ = g_bad.num_timesteps

        assert g.num_cells_p == 0
        assert g.num_cells_r == 2
        assert g.volumes is not None
        assert g.face_areas is not None

    def test_volumes_and_areas_at_the_origin(self):
        # The first face of a grid centered at the origin lies at -dr / 2, below r = 0
        g = Grid(r_centers=np.linspace(0.0, 4.0, 5) * su.LENGTH)
        assert g.r_faces[0] < 0.0 * su.LENGTH
        volumes = g.volumes.to_value(su.VOLUME)
        np.testing.assert_allclose(volumes[0], 4.0 * np.pi / 3.0 * 0.5**3, rtol=1e-15)
        np.testing.assert_allclose(
            volumes.sum(), 4.0 * np.pi / 3.0 * 4.5**3, rtol=1e-15
        )
        # The first face is the origin itself, and the others are spheres
        areas = g.face_areas.to_value(su.AREA)
        assert areas[0] == 0.0
        np.testing.assert_allclose(areas[1:], 4.0 * np.pi * (np.arange(5) + 0.5) ** 2)

    def test_post_process(self):
        g = Grid(p_centers=[50.0, 100.0] * su.MOMENTUM, is_p_log=True)
        assert np.isclose(g.p_centers[0], np.log10(50.0))
        g.post_process_calculations()
        assert np.isclose(g.p_centers[0].to_value(su.MOMENTUM), 50.0)

        # Non-log doesn't mutate
        g2 = Grid(p_centers=[50.0, 100.0] * su.MOMENTUM, is_p_log=False)
        g2.post_process_calculations()
        assert np.isclose(g2.p_centers[0].to_value(su.MOMENTUM), 50.0)

    def test_compatible_array(self):
        g = Grid(r_centers=[1.0, 2.0, 3.0] * su.LENGTH)
        arr = np.zeros(3)
        assert g.is_compatible_array(arr)
        arr2 = np.zeros((2, 3))
        assert not g.is_compatible_array(arr2)

    def test_four_pi_p2(self):
        # 1. Without momentum grid
        g_r = Grid(r_centers=[1.0, 2.0] * su.LENGTH)
        assert g_r.four_pi_p2 is None

        # 2. With non-log momentum grid
        p_c = np.array([2.0, 5.0]) * su.MOMENTUM
        g_p_lin = Grid(p_centers=p_c, is_p_log=False)
        np.testing.assert_allclose(
            g_p_lin.four_pi_p2.to_value(su.MOMENTUM**2),
            4.0 * np.pi * (np.array([2.0, 5.0]) ** 2),
        )

        # 3. With log momentum grid (should use physical coordinates)
        g_p_log = Grid(p_centers=p_c, is_p_log=True)
        np.testing.assert_allclose(
            g_p_log.four_pi_p2.to_value(su.MOMENTUM**2),
            4.0 * np.pi * (np.array([2.0, 5.0]) ** 2),
        )

    def test_str_representation(self):
        g = Grid(
            r_centers=[1.0, 2.0] * su.LENGTH,
            p_centers=[1.0, 2.0] * su.MOMENTUM,
            is_p_log=True,
            t_grid=[0, 1] * su.TIME,
        )
        s = str(g)
        assert "Grid:" in s
        assert "Spatial range:" in s
        assert "Momentum range:" in s
        assert "Temporal range:" in s


def test_physical_momentum_coordinates():
    # Linear momentum grids store physical momenta directly
    linear = Grid(p_centers=np.linspace(1.0, 5.0, 5) * su.MOMENTUM, is_p_log=False)
    np.testing.assert_allclose(
        linear.p_faces_phys.to_value(su.MOMENTUM), [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    )
    assert linear.p_centers_phys is linear.p_centers

    # Logarithmic grids store log10(p) but expose the physical momenta
    log = Grid(p_faces=np.array([1.0, 10.0, 100.0]) * su.MOMENTUM)
    np.testing.assert_allclose(log.p_faces, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        log.p_faces_phys.to_value(su.MOMENTUM), [1.0, 10.0, 100.0]
    )
