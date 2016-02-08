### From http://matplotlib.org/examples/animation/moviewriter.html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

def movie(X, Y):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='MWLMCH6', artist='Matplotlib',
               comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure()
    l, = plt.scatter([], [], 'k-o')
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    x0, y0 = 0, 0

    with writer.saving(fig, "writer_test.mp4", 100):
        for i in range(len(X)):
            x0 = X[i]
            y0 = Y[i]
            l.set_data(x0, y0)
            writer.grab_frame()
