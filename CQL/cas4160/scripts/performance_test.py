#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import gymnasium as gym
from cas4160.agents.cql_agent import CQLAgent
from pointmass import Pointmass
from cas4160.infrastructure import pytorch_util as ptu

def evaluate(agent_ckpt_path, difficulty, num_episodes=100):
    device = ptu.device
    # 1) Load the checkpoint
    ckpt = torch.load(agent_ckpt_path, map_location=device)
    # 2) Reconstruct the agent with the same params
    agent_params = ckpt['agent_params']
    agent = CQLAgent(**agent_params)
    # 3) Load Q‐network & target Q‐network weights
    agent.q_net.load_state_dict(ckpt['q_net_state_dict'])
    agent.q_net_target.load_state_dict(ckpt['q_net_target_state_dict'])
    agent.q_net.to(device)
    agent.q_net_target.to(device)
    # 4) If you have an actor network, load its weights too
    if 'actor_state_dict' in ckpt:
        agent.actor.load_state_dict(ckpt['actor_state_dict'])
        agent.actor.to(device)
    agent.eval()  # disable any training‐only behavior

    # 5) Create the Pointmass env with the same difficulty
    env = Pointmass(difficulty=difficulty, dense_reward=False)
    returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            # 6) Select action via the learned policy (greedy eval)
            obs_tensor = ptu.from_numpy(obs).unsqueeze(0)   # shape [1,2]
            with torch.no_grad():
                q_values = agent.q_net(obs_tensor)         # [1, ac_dim]
                action = int(q_values.argmax(dim=1).item())
            # 7) Step the env
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)
        print(f"Episode {ep+1:3d} return = {ep_return:.3f}")

    returns = np.array(returns)
    print(f"\n=== Evaluation over {num_episodes} episodes ===")
    print(f"Mean Return: {returns.mean():.4f}")
    print(f"Std  Return: {returns.std():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CQLAgent on 100 Pointmass episodes")
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the agent checkpoint (.pt) file")
    parser.add_argument(
        "--difficulty",
        type=int,
        default=2,
        help="Pointmass difficulty (0=Easy,...,3=VeryHard)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate(args.ckpt, args.difficulty, args.episodes)