import numpy as np
import pylab as plt
from numpy import exp


def logo(seed=4):
    trace_color = '#002288'
    text_color = 'green'
    ndets = 15
    rise_time = 3.
    fall_time = 11.0
    ph_other = 0.4
    t = np.arange(-15, 113, .1)
    x = exp(-t/fall_time)-exp(-t/rise_time)
    normalize = x.max()
    x[t <= 0] = 0

    # Second pulse
    t2 = 80
    x[t > t2] += .35*(exp(-(t[t > t2]-t2)/fall_time)-exp(-(t[t > t2]-t2)/rise_time))
    x /= normalize

    fig = plt.figure(9, figsize=(1.28, 1.28), dpi=100)
    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.85)
    plt.clf()
    plt.plot(t, x, color=trace_color, lw=2)
    plt.xticks([])
    plt.yticks([])
    plt.text(105, .65, 'Mass', ha='right', size=18, color=text_color)

    # Other dets
    np.random.seed(seed)
    cm = plt.cm.spectral
    for i in range(ndets):
        n = np.random.poisson(.8, size=1)
        x = np.zeros_like(t)
        for _ in range(n):
            t0 = np.random.uniform(-25, 110)
            print(t0,)
            x[t > t0] += exp(-(t[t > t0]-t0)/fall_time)-exp(-(t[t > t0]-t0)/rise_time)
        x *= ph_other/normalize
        plt.plot(t, x-.1*i-.35, color=cm(i/(ndets-.5)))
        print

    plt.ylim([-.4-.1*ndets, 1.2+.03*ndets])
