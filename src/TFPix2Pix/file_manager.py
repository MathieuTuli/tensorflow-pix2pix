import matplotlib.pyplot as plt
import numpy as np


def save_pyplot(file_name: str, image: np.ndarray) -> None:
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    plt.savefig(str(file_name))
