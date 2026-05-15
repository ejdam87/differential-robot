"""Particle resampling functionality."""

import numpy as np
from lib.types import Particle


def effective_sample_size(particles: list[Particle]) -> float:
    """
    Computes effective sample size (used as a threshold for resampling).

    params:
        particles: list of particles
    returns:
        effective sample size
    """
    total = 0
    for _, _, _, weight in particles:
        total += weight**2

    # more balanced weights ~ bigger value
    return 1 / total


def stochastic_universal_resampling(particles: list[Particle]) -> list[Particle]:
    """
    Resample particles using SUR algorithm.

    params:
        particles: list of particles
    returns:
        new particles
    """
    N = len(particles)
    if N == 0:
        return []

    # assuming normalized weights
    weights = np.array([w for _, _, _, w in particles])
    cdf = np.cumsum(weights)

    # Step size and random start
    step = 1.0 / N
    start = np.random.uniform(0, step)
    pointers = start + step * np.arange(N)

    new_particles = []
    i = 0
    for p in pointers:
        while p > cdf[i]:
            i += 1
        x, y, angle, _ = particles[i]
        new_particles.append((x, y, angle, 1.0 / N))

    return new_particles
