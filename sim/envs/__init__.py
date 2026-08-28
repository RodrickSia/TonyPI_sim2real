from gymnasium.envs.registration import register

from .flat_env import TonyPiFlatEnv

register(id="TonyPiFlat-v0", entry_point=TonyPiFlatEnv, max_episode_steps=1000)
