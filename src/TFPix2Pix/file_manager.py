import matplotlib.pyplot as plt
import numpy as np
import logging


def save_pyplot(file_name: str, image: np.ndarray) -> None:
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.canvas.draw()
    fig.add_axes(ax)
    ax.imshow(image, aspect='equal')
    plt.savefig(str(file_name), bbox_inches='tight')
    logging.info(f"TFPix2Pix: Save Pyplot: Image saved to {file_name}")
    plt.close(fig)
