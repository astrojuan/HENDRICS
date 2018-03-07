from stingray.events import EventList
from stingray.lightcurve import Lightcurve
import numpy as np
from ..ffa import ffa_search
from ..efsearch import fit


def test_ffa():
    period = 0.01
    pmin = 0.009
    pmax = 0.011
    dt = 10**int(np.log10(period)) / 256
    length = 10
    times = np.arange(0, length, dt)

    flux = 10 + np.cos(2 * np.pi * times / period)

    lc_cont = Lightcurve(times, flux, err_dist='gauss')

    ev = EventList()
    ev.simulate_times(lc_cont)
    lc = Lightcurve.make_lightcurve(ev.time, dt=dt, tstart=0, tseg=length)

    per, st = ffa_search(lc.counts, dt, pmin, pmax, nave=1)
    #  fit_sinc wants frequencies, not periods
    model = fit(1/per[::-1], st[::-1], 1/period, obs_length=10)
    assert np.isclose(1/model.mean, period, atol=1e-6)