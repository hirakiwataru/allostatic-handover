"""Registration helpers for robosuite / human-robot-gym."""

from __future__ import annotations


def register_robosuite_env() -> str:
    """Register the allostatic handover env with robosuite.

    Importing human-robot-gym registers the original environments. This helper
    adds the external subclass without editing the human-robot-gym repository.
    """
    try:
        from robosuite.environments.base import REGISTERED_ENVS
    except ImportError as exc:
        raise ImportError(
            "robosuite is required for backend='hrgym'. Activate the conda env "
            "from README.md or install human-robot-gym first."
        ) from exc

    from allostatic_handover.envs.allostatic_robot_human_handover import (
        AllostaticRobotHumanHandoverCart,
    )
    import human_robot_gym.robots  # noqa: F401

    REGISTERED_ENVS["AllostaticRobotHumanHandoverCart"] = AllostaticRobotHumanHandoverCart
    return "AllostaticRobotHumanHandoverCart"


def make_hrgym_env(**kwargs):
    """Create the real human-robot-gym allostatic environment."""
    env_name = register_robosuite_env()
    import robosuite

    kwargs.setdefault("robots", "Schunk")
    return robosuite.make(env_name, **kwargs)


def make_original_handover_env(**kwargs):
    """Create the original human-robot-gym RobotHumanHandoverCart environment."""
    try:
        import robosuite
    except ImportError as exc:
        raise ImportError(
            "robosuite is required for the original human-robot-gym handover env. "
            "Activate the conda env from README.md."
        ) from exc

    import human_robot_gym.robots  # noqa: F401
    import human_robot_gym.environments.manipulation.robot_human_handover_cartesian_env  # noqa: F401

    kwargs.setdefault("robots", "Schunk")
    return robosuite.make("RobotHumanHandoverCart", **kwargs)
