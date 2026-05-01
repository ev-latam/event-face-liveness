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

