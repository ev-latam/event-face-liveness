import numpy as np


def activity_profile(events, scale=1/10, tau=30e3):
    last_on =  -1
    last_off = -1
    at_on = []
    at_off  = []
    ts_on = []
    ts_off = []
    for (t,x,y,p) in events:
        if p > 0:
            if last_on == -1:
                last_on = t
                at_on.append(1)
            dt = (t - last_on) / tau
            at_on.append(at_on[-1]*np.exp(-dt) + 1/scale)
            ts_on.append(t)
            last_on = t
        else:
            if last_off == -1:
                last_off = t
                at_off.append(1)
            dt = (t - last_off) / tau
            at_off.append(at_off[-1]*np.exp(-dt) + 1/scale)
            ts_off.append(t)
            last_off = t 
    return at_on, at_off, ts_on, ts_off



def time_surface(events, size, t_ref, tau=6e3):

    ts, x, y, p = events

    H, W = size

    sae_on  = np.zeros((H, W), np.float32)
    sae_off = np.zeros((H, W), np.float32)

    if len(ts) == 0:
        return np.stack([sae_on, sae_off])

    decay = np.exp(-(t_ref - ts) / tau)

    on_mask  = p > 0
    off_mask = p <= 0

    sae_on[y[on_mask],  x[on_mask]]  = decay[on_mask]
    sae_off[y[off_mask], x[off_mask]] = decay[off_mask]

    return np.stack([sae_on, sae_off])