# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton 原生场景：运动学台面、水壶和目标容器。"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx


TABLE_HALF_EXTENTS = (1.6, 1.6, 0.05)
TABLE_POSITION = (0.0, 0.0, -0.05)
JUG_USD_PATH = Path("/home/magengyu/IsaacLab-Arena/scene/fstylejug_a01/fstylejug_a01_inst_physx.usd")
CONTAINER_USD_PATH = Path(__file__).with_name("open_cardboard_box_hydroelastic.usda")
JUG_POSITION = (0.45, -0.25, 0.01)
CONTAINER_POSITION = (0.45, 0.22, 0.02)
MESH_SDF_MAX_RESOLUTION = 256
MESH_SDF_NARROW_BAND_RANGE = (-0.04, 0.04)
MESH_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_sdf_cache"
TABLE_SHAPE_MARGIN = 0.005
TABLE_COLLISION_GAP = 0.01
# 水壶最小包围盒尺度约为 55 mm，2 mm 约占其 3.6%。
JUG_SHAPE_MARGIN = 0.002
JUG_COLLISION_GAP = 0.002
# 水密开口纸箱的外尺寸为 400 x 300 x 120 mm，壁厚为 4 mm。
CONTAINER_SHAPE_MARGIN = 0.002
CONTAINER_COLLISION_GAP = 0.010
CONTACT_STIFFNESS = 250000.0
CONTACT_DAMPING = 1000.0
# Hydroelastic uses pressure per signed-distance depth, so this is independent
# of the point-contact stiffness used by the table's conventional collider.
HYDROELASTIC_CONTACT_STIFFNESS = 1.0e10
RIGID_CONTACT_MAX = 4096
SIMULATION_FREQUENCY_HZ = 960


def _build_sdf_collision_models(
    builder: newton.ModelBuilder,
    asset_name: str,
    shape_indices: list[int],
    *,
    is_hydroelastic: bool,
) -> None:
    """Cook cached SDFs and optionally enable hydroelastic contact for closed meshes."""
    assert shape_indices, f"{asset_name} USD 未导入任何碰撞 shape。"
    for shape_index in shape_indices:
        geometry = builder.shape_source[shape_index]
        assert hasattr(geometry, "build_sdf"), f"{asset_name} shape={shape_index} 不是可构造 SDF 的网格。"
        _exit_if_mesh_is_not_watertight(geometry, asset_name, shape_index)
        geometry.build_sdf(
            max_resolution=MESH_SDF_MAX_RESOLUTION,
            narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
            margin=builder.shape_gap[shape_index],
            scale=tuple(float(value) for value in builder.shape_scale[shape_index]),
            cache_dir=MESH_SDF_CACHE_DIR,
        )
        assert geometry.sdf is not None, f"{asset_name} shape={shape_index} 的 SDF 构造失败。"
        builder.shape_force_sdf[shape_index] = True
        if is_hydroelastic:
            builder.shape_flags[shape_index] |= int(newton.ShapeFlags.HYDROELASTIC)
            builder.shape_material_kh[shape_index] = HYDROELASTIC_CONTACT_STIFFNESS


def _exit_if_mesh_is_not_watertight(geometry: newton.Mesh, asset_name: str, shape_index: int) -> None:
    """Exit when a collision mesh has boundary or non-manifold edges."""
    vertices = np.asarray(geometry.vertices)
    indices = np.asarray(geometry.indices, dtype=np.int64)
    assert indices.size % 3 == 0, f"{asset_name} shape={shape_index} 的索引数不是三角形数的整数倍。"
    assert indices.size == 0 or indices.max() < len(vertices), f"{asset_name} shape={shape_index} 的索引超出顶点范围。"
    # USD exporters often split vertices at UV or normal seams.  Weld only
    # exactly equal positions so those seams do not appear as open boundaries.
    _, welded_vertex_indices = np.unique(vertices, axis=0, return_inverse=True)
    triangles = welded_vertex_indices[indices].reshape(-1, 3)
    is_degenerate = np.logical_or.reduce(
        (triangles[:, 0] == triangles[:, 1], triangles[:, 1] == triangles[:, 2], triangles[:, 2] == triangles[:, 0])
    )
    degenerate_triangle_count = int(np.count_nonzero(is_degenerate))
    triangles = triangles[~is_degenerate]
    edges = triangles[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    edges.sort(axis=1)
    _, edge_use_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edge_count = int(np.count_nonzero(edge_use_counts == 1))
    non_manifold_edge_count = int(np.count_nonzero(edge_use_counts > 2))
    if boundary_edge_count == 0 and non_manifold_edge_count == 0 and degenerate_triangle_count == 0:
        return

    print(
        f"[ERROR] {asset_name} shape={shape_index} 不是水密碰撞网格："
        f"边界边={boundary_edge_count}，非流形边={non_manifold_edge_count}，"
        f"退化三角形={degenerate_triangle_count}。"
        "Hydroelastic 仅支持封闭体；请提供水密碰撞代理后重试。",
        flush=True,
    )
    raise SystemExit(1)


def _set_shape_contact_properties(
    builder: newton.ModelBuilder,
    shape_indices: list[int],
    margin: float,
    gap: float,
) -> None:
    """Set per-shape contact distance, stiffness, damping, and restitution."""
    for shape_index in shape_indices:
        builder.shape_margin[shape_index] = margin
        builder.shape_gap[shape_index] = gap
        builder.shape_material_ke[shape_index] = CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = CONTACT_DAMPING
        builder.shape_material_restitution[shape_index] = 0.0


def _add_hydroelastic_contact_copies(
    builder: newton.ModelBuilder, asset_name: str, shape_indices: list[int]
) -> None:
    """Add invisible hydroelastic copies while the source shapes remain hard-SDF guards."""
    for shape_index in shape_indices:
        geometry = builder.shape_source[shape_index]
        assert isinstance(geometry, newton.Mesh), f"{asset_name} shape={shape_index} 不是网格。"
        hydroelastic_shape = builder.add_shape_mesh(
            body=builder.shape_body[shape_index],
            xform=builder.shape_transform[shape_index],
            mesh=geometry,
            scale=tuple(float(value) for value in builder.shape_scale[shape_index]),
            cfg=newton.ModelBuilder.ShapeConfig(
                margin=builder.shape_margin[shape_index],
                gap=builder.shape_gap[shape_index],
                mu=builder.default_shape_cfg.mu,
                ke=CONTACT_STIFFNESS,
                kd=CONTACT_DAMPING,
                restitution=0.0,
                is_visible=False,
                force_sdf=True,
                is_hydroelastic=True,
                kh=HYDROELASTIC_CONTACT_STIFFNESS,
            ),
            label=f"{asset_name}_hydroelastic",
        )
        assert builder.shape_force_sdf[hydroelastic_shape], f"{asset_name} hydroelastic 副本未启用 SDF。"


class Example:
    """Build and simulate the native Newton scene."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.log_every = args.log_every
        self.step_count = 0
        self.frame_dt = 1.0 / 120.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.rigid_gap = TABLE_COLLISION_GAP
        builder.default_shape_cfg.mu = 1.0
        table_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(*TABLE_POSITION), q=wp.quat_identity()),
            is_kinematic=True,
            label="table",
        )
        table_shape = builder.add_shape_box(
            body=table_body,
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            color=wp.vec3(0.2, 0.2, 0.2),
            label="table",
        )
        builder.shape_margin[table_shape] = TABLE_SHAPE_MARGIN
        builder.shape_gap[table_shape] = TABLE_COLLISION_GAP
        builder.shape_material_ke[table_shape] = CONTACT_STIFFNESS
        builder.shape_material_kd[table_shape] = CONTACT_DAMPING
        builder.shape_material_restitution[table_shape] = 0.0

        jug_result = builder.add_usd(
            str(JUG_USD_PATH),
            xform=wp.transform(p=wp.vec3(*JUG_POSITION), q=wp.quat_identity()),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        jug_bodies = list(jug_result["path_body_map"].values())
        assert len(jug_bodies) == 1, "水壶 Physics USD 应当只包含一个刚体。"
        self.jug_body = jug_bodies[0]
        jug_shape_indices = list(jug_result["path_shape_map"].values())
        _set_shape_contact_properties(builder, jug_shape_indices, JUG_SHAPE_MARGIN, JUG_COLLISION_GAP)
        _build_sdf_collision_models(builder, "水壶", jug_shape_indices, is_hydroelastic=True)

        container_result = builder.add_usd(
            str(CONTAINER_USD_PATH),
            xform=wp.transform(p=wp.vec3(*CONTAINER_POSITION), q=wp.quat_identity()),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        container_shape_indices = list(container_result["path_shape_map"].values())
        _set_shape_contact_properties(
            builder,
            container_shape_indices,
            CONTAINER_SHAPE_MARGIN,
            CONTAINER_COLLISION_GAP,
        )
        # Keep a hard SDF copy for high-speed anti-tunnelling.  The invisible
        # duplicate retains hydroelastic pressure contact for slow interactions.
        _build_sdf_collision_models(builder, "目标容器", container_shape_indices, is_hydroelastic=False)
        _add_hydroelastic_contact_copies(builder, "目标容器", container_shape_indices)

        self.model = builder.finalize()
        self.model.rigid_contact_max = RIGID_CONTACT_MAX
        sdf_hydroelastic_config = HydroelasticSDF.Config(
            # Avoid biasing the marching-cubes contact surface near its edges.
            mc_edge_clamp_min=0.0,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=RIGID_CONTACT_MAX,
            broad_phase="sap",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=RIGID_CONTACT_MAX,
            nconmax=RIGID_CONTACT_MAX,
            iterations=100,
            ls_iterations=50,
            ccd_iterations=35,
            impratio=1.0,
            cone="pyramidal",
            tolerance=1.0e-6,
            use_mujoco_cpu=False,
            use_mujoco_contacts=False,
            update_data_interval=1,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.0, -2.0, 1.5), pitch=-15.0, yaw=135.0)
        self.graph = self._capture()

    def _simulate(self) -> None:
        """Advance one displayed frame using MuJoCo-Warp substeps."""
        for _ in range(self.sim_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self):
        """Capture the simulation step when the active device supports it."""
        if not wp.get_device().is_cuda:
            return None
        try:
            with wp.ScopedCapture() as capture:
                self._simulate()
            return capture.graph
        except Exception as exc:
            warnings.warn(f"CUDA graph capture failed; using eager stepping: {exc}", stacklevel=2)
            return None

    def step(self) -> None:
        """Advance the simulation and print the jug world position."""
        if self.graph is None:
            self._simulate()
        else:
            wp.capture_launch(self.graph)
        self.step_count += 1
        self.sim_time += self.frame_dt
        if self.step_count % self.log_every == 0:
            position = self.state_0.body_q.numpy()[self.jug_body, :3]
            print(f"[INFO] step={self.step_count:04d}, 水壶世界坐标: {position}", flush=True)

    def render(self) -> None:
        """Render the current Newton state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--log-every", type=int, default=1000, help="每隔多少帧打印一次水壶位置。")
    viewer, args = newton.examples.init(parser)
    assert args.log_every > 0, "--log-every 必须为正整数。"
    newton.examples.run(Example(viewer, args), args)
