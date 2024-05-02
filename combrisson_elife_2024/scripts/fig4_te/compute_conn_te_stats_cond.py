"""Compute Transfer Entropy per condition + CI + stats.
"""
import os

import numpy as np
import xarray as xr
import pandas as pd
import argparse
from joblib import Parallel, delayed

from frites.io import logger
from frites.utils import nonsorted_unique
from frites.workflow import WfStats
from frites.core import copnorm_nd
from frites.config import CONFIG
from frites.stats import confidence_interval
from frites.conn import conn_links
from frites.conn.conn_te import _para_te

from pathta import Study

from research.study.pblt import get_hga


###############################################################################
###############################################################################
#                                    CORE
###############################################################################
###############################################################################


def compute_te(x, net, biascorrection, min_delay, step_delay, max_delay,
               n_perm, n_jobs, hemi, hemi_links):
    """Compute FIT and FIT_p for a specific condition, for a single subject.
    """
    # __________________________________ I/O __________________________________
    # get coordinates
    roi, times = x['roi'].data, x['times'].data
    n_trials = len(x['trials'])

    # transpose x to avoid shape checking (n_roi, n_times, 1, n_trials)
    x = x.data.transpose(1, 2, 0)[..., np.newaxis, :]

    # define delay range and new time vector
    delays = np.arange(-max_delay, -min_delay, step_delay)[::-1]
    times_te = times[max_delay::]

    # copnorm (only for significance testing)
    x = copnorm_nd(x, axis=-1)

    # get links to compute
    (x_s, x_t), roi_st = conn_links(
        roi, directed=True, net=net, roi_relation='inter', hemisphere=hemi,
        hemi_links=hemi_links, verbose=True
    )

    # generate partitions
    part = []
    for k in range(n_perm):
        state = np.random.RandomState(seed=k)
        _part = np.arange(n_trials)
        state.shuffle(_part)
        part.append(_part)

    # __________________________________ FIT __________________________________
    logger.info("    - Compute TE")
    CONFIG["KW_GCMI"]["biascorrect"] = bool(biascorrection)
    te = _compute_te(x, x_s, x_t, max_delay, delays, None)

    logger.info("    - Compute permuted TE")
    CONFIG["KW_GCMI"]["biascorrect"] = bool(biascorrection)
    te_p = Parallel(n_jobs=n_jobs)(delayed(_compute_te)(
        x, x_s, x_t, max_delay, delays, p) for p in part)
    te_p = np.stack(te_p, axis=0)

    # ________________________________ XARRAY _________________________________
    logger.info("    - Formatting")
    te = xr.DataArray(te, dims=('roi', 'times'), coords=(roi_st, times_te))
    te_p = xr.DataArray(te_p, dims=('perm', 'roi', 'times'),
                         coords=(np.arange(n_perm), roi_st, times_te))

    return te, te_p


def _compute_te(x, x_s, x_t, max_delay, delays, part):
    """Compute single TE.
    """
    _, n_pts, _, _ = x.shape
    te = np.zeros((len(x_s), n_pts - max_delay), dtype=float)
    for n_p, (s, t) in enumerate(zip(x_s, x_t)):
        # select source / target signals
        _xs, _xt = x[s, ...], x[t, ...]

        # trial shuffling the source
        if isinstance(part, np.ndarray):
            _xs = _xs[..., part]

        # compute te
        te[n_p, :] = _para_te(_xs, _xt, max_delay, False, delays)
    return te


def compute_stat(name, x, x_p, times, roi, **kw_stats):
    """Compute the statistics.
    """
    # define xarray dims and coords
    dims, coords = ('times', 'roi'), (times, roi)
    kw_xr = dict(dims=dims, coords=coords)

    # compute mi effect size
    mi = np.stack([k.mean(0) for k in x], axis=-1)
    __dt = {
        f'mi_{name}': xr.DataArray(mi, name='MI', **kw_xr)
    }

    # compute confidence interval
    cis = [90, 95, 99, 'sd', 'sem']
    kw_ci = dict(axis=0, n_boots=1000, random_state=0, cis=cis)
    ci = np.stack([confidence_interval(k, **kw_ci) for k in x], axis=-1)
    __dt[f"ci_{name}"] = xr.DataArray(
        ci, dims=('ci', 'bound', 'times', 'roi'),
        coords=(cis, ['low', 'high'], times, roi)
    )

    # compute stats
    for sigma in [0., 0.001, 0.01, 0.1]:
        pv, tv = WfStats(verbose=False).fit(x, x_p, rfx_sigma=sigma,
                                            **kw_stats)
        # fill the dict
        sstr = "%.3f" % sigma
        __dt[f"pv_{name}_{sstr[2:]}"] = xr.DataArray(
            pv, name='p-values', **kw_xr)
        __dt[f"tv_{name}_{sstr[2:]}"] = xr.DataArray(
            tv, name='t-values', **kw_xr)

    return __dt


if __name__ == '__main__':
    ###########################################################################
    # filtering settings
    savgol = 10  # {float, None}

    # stat settings
    inference = 'rfx'
    mcp = 'cluster'

    # i/o settings
    to_folder = f'conn/te/stats/cond'
    ###########################################################################


    st = Study('PBLT')
    st.runtime()
    subjects = list(st.load_config('subjects.json').keys())

    # ________________________________ INPUTS _________________________________
    __bands = ['delta', 'theta', 'alpha', 'beta', 'lga', 'hga', 'bga', 'vlga']
    parser = argparse.ArgumentParser(description='Compute TE')
    parser.add_argument('--mindelay', default=5, type=int, help='Min delay')
    parser.add_argument('--maxdelay', default=10, type=int, help='Max delay')
    parser.add_argument('--stepdelay', default=2, type=int, help='Step delay')
    parser.add_argument('--hemilinks', default='intra', type=str,
                        choices=['both', 'inter', 'intra'], help='Hemi links')
    parser.add_argument('--band', default='lga', type=str,
                        choices=__bands, help='Power band')
    parser.add_argument('--net', default=0, type=int,
                        choices=[0, 1], help='Net FIT')
    parser.add_argument('--ds', default=0, type=int,
                        choices=[0, 2, 3, 4], help='Downsampling factor')
    parser.add_argument('--nperm', default=200, type=int, help='Permutations')
    parser.add_argument('--correction', default=1, type=int,
                        choices=[0, 1], help='Baseline correction')
    parser.add_argument('--biascorrection', default=1, type=int,
                        choices=[0, 1], help='Bias correction')
    args = parser.parse_args()
    min_delay = int(args.mindelay)
    max_delay = int(args.maxdelay)
    step_delay = int(args.stepdelay)
    hemi_links = str(args.hemilinks)
    band = str(args.band)
    net = int(args.net)
    ds = int(args.ds)
    n_perm = int(args.nperm)
    correction = int(args.correction)
    biascorrection = int(args.biascorrection)

    # set bias correction
    CONFIG["KW_GCMI"]["biascorrect"] = bool(biascorrection)

    # ________________________________ CONFIG _________________________________
    logger.info('=' * 79)
    logger.info(f"BAND={band}; NET={net}")
    logger.info(f"BASELINE CORRECTION={bool(correction)}")
    logger.info(f"BIAS CORRECTION={bool(biascorrection)}")
    logger.info(f"DOWN-SAMPLING FACTOR={ds}")
    logger.info(f"INFERENCE={inference};MCP={mcp};NPERM={n_perm}")
    logger.info('=' * 79)

    # __________________________________ I/O __________________________________
    # savgol string
    savstr = 'none' if not isinstance(savgol, (int, float)) else str(savgol)
    # define how to save the file
    _save_as = (
        f"te-{band}_delay_{min_delay}-{step_delay}-{max_delay}_"
        f"net-{net}_correction-{correction}_bias-{biascorrection}_"
        f"savgol-{savstr}_nperm-{n_perm}_inference-{inference}_mcp-{mcp}_"
        f"ds-{ds}_{hemi_links}-hemi.nc"
    )
    # define the template for saving the file
    save_as = st.join(_save_as, folder=to_folder, force=True)

    # skip if already computed
    if os.path.isfile(save_as):
        logger.warning('------------ ALREADY COMPUTED. SKIPPED. ------------')
        exit()

    # __________________________________ FIT __________________________________

    te, te_p = [], []
    for s in subjects:
        logger.info(f"-> Processing subject {s}")

        # -------------------------------- HGA --------------------------------
        # load the hga
        out = get_hga(s, band=band, savgol=savgol, ds=ds, is_conn=True,
                      correction=correction)
        if out is None: continue

        # unwrap arguments
        _da, sfreq, hemi = out['data'], out['sfreq'], out['hemi']
        _da['trials'] = out['cond']

        # --------------------------------- TE --------------------------------
        _te, _te_p = {}, {}
        for n_c, c in enumerate(['rew', 'pun']):
            _te[c], _te_p[c] = compute_te(
                _da.sel(trials=c), net, biascorrection, min_delay, step_delay,
                max_delay, n_perm, -1, hemi, hemi_links
            )
        te.append(xr.Dataset(_te).to_array('cond'))
        te_p.append(xr.Dataset(_te_p).to_array('cond'))

        del _te, _te_p

    ###########################################################################
    #                              REORGANIZE
    ###########################################################################
    """
    te = [(n_cond, n_roi, n_times), ...] * n_subjects
    te_p = [(n_cond, n_perm, n_roi, n_times), ...] * n_subjects
    """
    # get the unique list of brain regions
    roi_all = np.concatenate([te[k]['roi'].data for k in range(len(te))])
    roi_all = nonsorted_unique(roi_all)

    # reorganization
    roir, ter, ter_p = [], [], []
    for r in roi_all:
        # find the subjects having contacts in the pair of roi
        indices = {}
        for s in range(len(te)):
            if r in te[s]['roi'].data:
                indices[s] = np.where(te[s]['roi'].data == r)[0]

        # decide to concatenate (or not)
        _suj = indices.items()
        roir.append(r)
        ter.append(
            xr.concat([te[s].isel(roi=v) for s, v in _suj], 'roi'))
        ter_p.append(
            xr.concat([te_p[s].isel(roi=v) for s, v in _suj], 'roi'))
    del te, te_p

    ###########################################################################
    #                              FIT - STATS
    ###########################################################################
    kw_stats = dict(inference=inference, mcp=mcp)
    if net:
        kw_stats['tail'] = 0
    times = ter[0]['times'].data
    dt = {}

    # __________________________________ REW __________________________________
    logger.info('-> Stats : REW')
    te_rew = [k.sel(cond='rew').data for k in ter]
    te_p_rew = [k.sel(cond='rew').data for k in ter_p]
    _outs = compute_stat(
        'rew', te_rew, te_p_rew, times, roir, **kw_stats
    )
    dt.update(**_outs)

    # __________________________________ PUN __________________________________
    logger.info('-> Stats : PUN')
    te_pun = [k.sel(cond='pun').data for k in ter]
    te_p_pun = [k.sel(cond='pun').data for k in ter_p]
    _outs = compute_stat(
        'pun', te_pun, te_p_pun, times, roir, **kw_stats
    )
    dt.update(**_outs)

    if not net:
        # _____________________________ REW > PUN _____________________________
        logger.info('-> Stats : REW > PUN')
        te_rewpun = [r - p for r, p in zip(te_rew, te_pun)]
        te_p_rewpun = [r - p for r, p in zip(te_p_rew, te_p_pun)]
        _outs = compute_stat(
            'rew>pun', te_rewpun, te_p_rewpun, times, roir, **kw_stats
        )
        dt.update(**_outs)

        # _____________________________ PUN > REW _____________________________
        logger.info('-> Stats : PUN > REW')
        te_punrew = [p - r for r, p in zip(te_rew, te_pun)]
        te_p_punrew = [p - r for r, p in zip(te_p_rew, te_p_pun)]
        _outs = compute_stat(
            'pun>rew', te_punrew, te_p_punrew, times, roir, **kw_stats
        )
        dt.update(**_outs)

        # ____________________________ REW != PUN _____________________________
        logger.info('-> Stats : REW != PUN')
        _outs = compute_stat(
            'rew!=pun', te_rewpun, te_p_rewpun, times, roir, tail=0,
            **kw_stats
        )
        dt.update(**_outs)

    ###########################################################################
    #                                  EXPORT
    ###########################################################################
    dt = xr.Dataset(dt)
    dt.to_netcdf(save_as)

    st.runtime()
