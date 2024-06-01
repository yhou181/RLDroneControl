from __future__ import annotations

from typing import Any, Literal

import numpy as np

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv


class QuadXHoverTestEnv(QuadXBaseEnv):


    def __init__(
        self,
        sparse_reward: bool = False,
        flight_mode: int = 0,
        flight_dome_size: float = 3.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 40,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        start_pos : np.ndarray = np.array([[0., 0., 1.0]])
    ):

        super().__init__(
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
            start_pos=start_pos
        )

        """GYMNASIUM STUFF"""
        self.observation_space = self.combined_space


        self.sparse_reward = sparse_reward

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:

        super().begin_reset(seed, options)
        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self) -> None:

        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # combine everything
        if self.angle_representation == 0:
            self.state = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action, *aux_state]
            )
        elif self.angle_representation == 1:
            self.state = np.array(
                [*ang_vel, *quarternion, *lin_vel, *lin_pos, *self.action, *aux_state]
            )

    def compute_term_trunc_reward(self) -> None:
        super().compute_base_term_trunc_reward()

        if not self.sparse_reward:
            # distance from 0, 0, 1 hover point
            linear_distance = np.linalg.norm(
                self.env.state(0)[-1] - np.array([0.0, 0.0, 1.0])
            )

            # how far are we from 0 roll pitch
            angular_distance = np.linalg.norm(self.env.state(0)[1][:2])

            self.reward -= linear_distance + angular_distance
            self.reward += 1.0
