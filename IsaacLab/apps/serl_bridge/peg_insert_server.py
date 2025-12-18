"""Socket-based server exposing the IsaacLab peg insert environment to SERL."""

from __future__ import annotations

import argparse
import logging
import signal
from contextlib import closing
from multiprocessing.connection import Listener
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch

from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from .peg_insert_env import SerlFactoryPegInsertEnv

LOGGER = logging.getLogger("isaaclab.serl_bridge")

########################################################################################

########################################################################################
def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return array


def _format_obs(obs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    policy = _tensor_to_numpy(obs["policy"])  # shape (1, dim)
    critic = _tensor_to_numpy(obs["critic"])
    #image = _tensor_to_numpy(obs["camera"]).astype(np.uint8)
    return {
        "policy": policy[0],
        "critic": critic[0],
        #"image": image[0],
    }


def _format_info(info: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, torch.Tensor):
            array = _tensor_to_numpy(value)
            if array.shape == ():
                formatted[key] = array.item()
            elif array.shape[0] == 1:
                formatted[key] = array[0]
            else:
                formatted[key] = array
        else:
            formatted[key] = value
    return formatted


def _handle_client(env: SerlFactoryPegInsertEnv, conn, device: torch.device):
    while True:
        try:
            message = conn.recv()
        except EOFError:
            break

        command = message.get("cmd")
        if command == "reset":
            seed = message.get("seed")
            obs, info = env.reset(seed=seed)
            payload = {"obs": _format_obs(obs), "info": _format_info(info)}
            conn.send(payload)
        elif command == "step":
            action = np.asarray(message["action"], dtype=np.float32)
            action_tensor = torch.from_numpy(action).to(device).unsqueeze(0)
            obs, reward, terminated, truncated, info = env.step(action_tensor)

            payload = {
                "obs": _format_obs(obs),
                "reward": float(_tensor_to_numpy(reward)[0]),
                "terminated": bool(_tensor_to_numpy(terminated)[0]),
                "truncated": bool(_tensor_to_numpy(truncated)[0]),
                "info": _format_info(info),
            }
            conn.send(payload)
        elif command == "close":
            conn.send({"status": "ok"})
            break
        else:
            conn.send({"error": f"Unknown command: {command}"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP to bind the environment server.")
    parser.add_argument("--port", type=int, default=6000, help="Port to bind the environment server.")
    parser.add_argument(
        "--authkey",
        default="serl",
        help="Authentication key used for the Python multiprocessing connection.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    # Cameras are required for the SERL observations, so make sure they are enabled
    if not getattr(args, "enable_cameras", False):
        LOGGER.info("Forcing enable_cameras=True to stream RGB observations to SERL")
        setattr(args, "enable_cameras", True)

    # Launch the Isaac Lab application before importing any modules that rely on Omniverse extensions
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from peg_insert_env import SerlFactoryPegInsertEnv, SerlFactoryTaskPegInsertCfg

    cfg = SerlFactoryTaskPegInsertCfg()
    cfg.sim.device = args.device

    LOGGER.info("Creating IsaacLab peg insert environment on device %s", args.device)
    print("cfg is ", cfg)
    env = SerlFactoryPegInsertEnv(cfg=cfg)
    listener = Listener((args.host, args.port), authkey=args.authkey.encode())
    LOGGER.info("Environment server listening on %s:%d", args.host, args.port)

    stop_requested = False

    def _request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        LOGGER.info("Received signal %s. Shutting down after current episode.", signum)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        while not stop_requested:
            LOGGER.info("Waiting for SERL actor connection...")
            try:
                conn = listener.accept()
            except (OSError, EOFError):
                break
            LOGGER.info("Actor connected from %s", listener.last_accepted)
            with closing(conn):
                _handle_client(env, conn, torch.device(args.device))
            if stop_requested:
                break
    finally:
        LOGGER.info("Shutting down IsaacLab environment")
        listener.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()