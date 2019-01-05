import math
import numpy as np
import fire

from matplotlib import pyplot as plt
from pathlib import Path

data_dir = Path(__file__).parent / 'data'


#   returns a random point located on perimeter of circle with given radius

def random_point(radius):
    x = np.random.uniform(0.0, radius)
    y = math.sqrt(radius ** 2 - x ** 2)
    x *= 1.0 if np.random.randint(2) == 0 else -1.0
    y *= 1.0 if np.random.randint(2) == 0 else -1.0
    return x, y


#   returns a tuple of features and labels, shuffling required

def sample_circles(samples, inner_radius, outer_radius, noise):
    features = np.zeros((samples, 2))
    labels = np.zeros(samples)
    noise = abs(noise)

    #   generates samples for inner circle

    for i in range(samples // 2):
        (x, y) = random_point(abs(inner_radius + np.random.normal(0, noise)))
        features[i][0] = x
        features[i][1] = y
        labels[i] = 0

    #   generates samples for outer circle

    for i in range(samples // 2, samples):
        (x, y) = random_point(abs(outer_radius + np.random.normal(0, noise)))
        features[i][0] = x
        features[i][1] = y
        labels[i] = 1

    return features, labels


def polar_to_cartesian(angle, radii):
    x = radii * np.cos(angle)
    y = radii * np.sin(angle)
    return np.concatenate((x, y), axis=1)


def generate_circles(number_of_points: int=100000, inner_radius: float=0.65, outer_radius: float=1.0, save: bool=True):
    inner_points = number_of_points // 2
    outer_points = number_of_points - inner_points
    training_points = int(number_of_points / 1.25)

    angles = np.random.uniform(0, 2 * np.pi, (number_of_points, 1))
    noise = np.random.normal(0, 0.05, (number_of_points, 1))

    inner_radii = inner_radius + noise[:inner_points]
    outer_radii = outer_radius + noise[inner_points:]

    inner_circle = polar_to_cartesian(angles[:inner_points], inner_radii)
    outer_circle = polar_to_cartesian(angles[inner_points:], outer_radii)

    labels = np.concatenate((np.zeros((inner_points, 1)), np.ones((outer_points, 1))))
    features = np.concatenate((inner_circle, outer_circle))

    combined = np.concatenate((features, labels), axis=1)
    np.random.shuffle(combined)

    training = combined[:training_points]
    test = combined[training_points:]

    if save:
        np.save(str(data_dir / 'circle_training.npy'), training)
        np.save(str(data_dir / 'circle_test.npy'), test)
    else:
        fig, ax = plt.subplots()
        ax.plot(features[:inner_points, 0], features[:inner_points, 1], linestyle='None', marker='o')
        ax.plot(features[inner_points:, 0], features[inner_points:, 1], linestyle='None', marker='o')
        plt.show()


if __name__ == '__main__':
    fire.Fire(generate_circles)
