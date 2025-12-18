"""Custom IsaacLab Peg Insert environment with camera output for SERL."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg


@configclass
class SerlFactoryTaskPegInsertCfg(FactoryTaskPegInsertCfg):
    """Configuration for the SERL-compatible Peg Insert task.

    This variant forces a single-environment setup and attaches an RGB camera that
    roughly matches the wrist cameras used in the SERL examples.
    """

    # attach a tiled RGB camera looking towards the workspace
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.55, 0.0, 0.6),
            rot=(0.9238795, 0.0, 0.3826834, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        width=128,
        height=128,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=1.0,
            horizontal_aperture=20.0,
            clipping_range=(0.01, 3.0),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        # run a single environment instance since the actor expects a non-vectorized env
        self.scene = self.scene.replace(
            num_envs=1,
            env_spacing=2.0,
            clone_in_fabric=False,
        )
        # make sure rendering happens each control step so that camera images are valid
        self.sim = self.sim.replace(render_interval=self.decimation)


class SerlFactoryPegInsertEnv(FactoryEnv):
    """Factory Peg Insert task augmented with a tiled camera for SERL."""

    cfg: SerlFactoryTaskPegInsertCfg

    def __init__(self, cfg: SerlFactoryTaskPegInsertCfg, render_mode: str | None = None, **kwargs):
        self._camera: TiledCamera | None = None
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()
        # attach tiled camera after the rest of the scene is created
        self._camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["agent_camera"] = self._camera

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if self._camera is not None:
            self._camera.reset(env_ids)

    def _get_observations(self):
        obs = super()._get_observations()
        # if self._camera is not None:
        #     camera_data = self._camera.data.output["rgb"].clone()
        #     # keep RGB channels only and ensure data is contiguous
        #     obs["camera"] = camera_data[..., :3].contiguous()
        return obs