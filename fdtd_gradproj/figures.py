
import numpy as np
import matplotlib.pyplot as plt

def plot_field_snapshots(x, snapshots, labels=None, title=None, fname=None):
    plt.figure()
    for i, y in enumerate(snapshots):
        lab = None if labels is None else labels[i]
        plt.plot(x, y, label=lab)
    plt.xlabel("x [m]")
    plt.ylabel("Field (a.u.)")
    if labels is not None: plt.legend()
    if title: plt.title(title)
    if fname:
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_time_series(t, series, labels=None, title=None, fname=None):
    plt.figure()
    for i, y in enumerate(series):
        lab = None if labels is None else labels[i]
        plt.plot(t, y, label=lab)
    plt.xlabel("t [s]")
    plt.ylabel("Amplitude (a.u.)")
    if labels is not None: plt.legend()
    if title: plt.title(title)
    if fname:
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
