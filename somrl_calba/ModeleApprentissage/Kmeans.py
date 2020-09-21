#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import argparse


class Kmeans:

    def __init__(self, N, dim, epsilon):
        self.epsilon = epsilon
        self.N = N
        self.dim = dim
        self.init_weights()
	
    def init_weights(self):
        self.weights = 0.5 * (2 * np.random.random((self.N, self.dim)) - 1.0)

    def epochs(self, t, x):
        # Find the Best Matching Unit
        dist = np.linalg.norm(self.weights - x, axis=1)
        bmu_idx = np.argmin(dist)
        # Adapt its weights
        self.weights[bmu_idx] += self.epsilon * (x - self.weights[bmu_idx])


def sample_anneau(radius_in, radius_out):
    found = False
    while not found:
        x = radius_out * (2 * np.random.random((2, )) - 1.0)
        found = np.linalg.norm(x) >= radius_in and np.linalg.norm(x) <= radius_out
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--movie',
                        help='Should we generate a movie',
                        action='store_true')
    parser.add_argument('--fps', type=int, default=25,
                        help='Frame rate')
    parser.add_argument('--frames', type=int, default=1000,
                        help='Number of frames')
    args = parser.parse_args()

    # Extract the arguments
    save_movie = True
    frames = args.frames
    fps = args.fps

    if save_movie:
        matplotlib.use("Agg")

    # The distribution from which the samples will be drawn
    oracle = lambda: sample_anneau(0.5, 1.0)

    # The algorithm
    kmeans = Kmeans(N=20, dim=2, epsilon=1e-1)

    # Sample some points for drawing
    points = np.array([oracle() for i in range(10000)])

    # Setup the plot
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], s=4)
    proto = ax.scatter(kmeans.weights[:, 0], kmeans.weights[:, 1], s=20,
                       animated=True)
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    text = ax.annotate('0',
                       xy=(-0.95, -0.95),
                       fontsize=10)
    plt.gca().set_aspect('equal')

    def init():
        global kmeans
        kmeans.init_weights()
        return proto, text

    def updatefig(frame):
        global kmeans, text
        x = oracle()
        kmeans.epochs(frame, x)
        proto.set_offsets(kmeans.weights)
        text.set_text('Sample {}'.format(frame))
        return proto, text

    ani = animation.FuncAnimation(fig, updatefig, init_func=init,
                                  frames=frames, blit=True, interval=1000/fps)
    if save_movie:
        ani.save('kmeans.mp4', writer='ffmpeg')
    else:
        plt.show()
