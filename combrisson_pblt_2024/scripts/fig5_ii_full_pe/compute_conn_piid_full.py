"""Compute conn (PID | II) + stats across condition in the time domain.

This script can :
- Compute connectivity metrics like conn_pid and conn_ii
- Perform RFX-contacts on ii (two-tailed), redu | syn | redu - syn | syn - redu
  (one-tailed)
- Computations are performed within conditions (i.e. rew | pun)
- Significance testing | Confidence interval
"""
import os

import numpy as np
import xarray as xr
import pandas as pd
import argparse

from joblib import Parallel, delayed

from frites.io import logger, set_log_level
from xfrites.conn import conn_ii, conn_pid
from frites.config import CONFIG
from frites.utils import nonsorted_unique
from frites.core import copnorm_nd
from frites.workflow import WfStats
from frites.stats import confidence_interval

from pathta import Study

from research.study.pblt import (get_mi_type, get_hga)


###############################################################################
###############################################################################
#                           CONN + STATS FUNCTIONS
###############################################################################
###############################################################################

def _fcn_conn(x, metric, **kwargs):
    """Connectivity metric to compute.
    """
    if metric == 'ii':
        return conn_ii(x, **kwargs)[1].astype(np.float32)
    elif metric == 'redu':
        return conn_pid(x, **kwargs)[3].astype(np.float32)
    elif metric == 'syn':
        return conn_pid(x, **kwargs)[4].astype(np.float32)
    elif metric == 'redu-syn':
        _, _, _, _redu, _syn = conn_pid(x, **kwargs)
        return (_redu - _syn).astype(np.float32)
    elif metric == 'syn-redu':
        _, _, _, _redu, _syn = conn_pid(x, **kwargs)
        return (_syn - _redu).astype(np.float32)



def compute_conn(x, mi_type, metric, n_perm, n_jobs, hemi, hemi_links,
                 relation):
    # ______________________________ II SETTINGS ______________________________
    # define arguments for the conn_ii
    kw_ii = dict(roi='roi', times='times', mi_type=mi_type, verbose=False,
                 gcrn=False, hemisphere=hemi, hemi_links=hemi_links,
                 roi_relation=relation)
    n_tr, n_roi, n_times = x.shape

    # __________________________________ CONN _________________________________
    ii = _fcn_conn(x, metric, y='model', **kw_ii)

    # _______________________________ SIGNI / CI ______________________________
    # generate partitions (shuffle y variable)
    y_p = []
    for k in range(n_perm):
        state = np.random.RandomState(seed=k)
        _part = np.arange(n_tr)
        state.shuffle(_part)
        y_p.append(x['model'].data[_part])

    # compute permutations
    ii_p = Parallel(n_jobs=n_jobs)(delayed(_fcn_conn)(
        x, metric, y=p, **kw_ii) for p in y_p)

    # merge permutations
    ii_p = xr.concat(ii_p, 'perm')
    ii_p['perm'] = np.arange(n_perm)

    return ii, ii_p


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
        wf = WfStats(verbose=False)
        pv, tv = wf.fit(x, x_p, rfx_sigma=sigma, **kw_stats)
        attrs = wf.attrs.copy()
        # fill the dict
        sstr = "%.3f" % sigma
        __dt[f"pv_{name}_{sstr[2:]}"] = xr.DataArray(
            pv, name='p-values', attrs=attrs, **kw_xr)
        __dt[f"tv_{name}_{sstr[2:]}"] = xr.DataArray(
            tv, name='t-values', attrs=attrs, **kw_xr)

    return __dt


###############################################################################
###############################################################################
#                                    MAIN
###############################################################################
###############################################################################


if __name__ == '__main__':
    ###########################################################################
    # filtering settings
    savgol = 10  # {float, None}

    # stat settings
    n_perm = 1000
    mcp = 'cluster'
    inference = 'rfx'

    # i/o settings
    to_folder = f'conn/piid/%s-%s/%s/full'
    ###########################################################################

    st = Study('PBLT')
    st.runtime()
    subjects = list(st.load_config('subjects.json').keys())  # [0:2]

    # ================================= INPUTS ================================
    __bands = ['delta', 'theta', 'alpha', 'beta', 'lga', 'hga', 'bga', 'vlga']
    parser = argparse.ArgumentParser(description='Compute MI')
    parser.add_argument('--model', default='PE', type=str,
                        choices=['PE', 'outc', 'blocks', 'PC', 'surprise'],
                        help='Model-free of model-based analysis')
    parser.add_argument('--metric', default='ii', type=str,
                        choices=['ii', 'redu', 'syn', 'redu-syn', 'syn-redu'],
                        help='Connectivity metric')
    parser.add_argument('--hemilinks', default='intra', type=str,
                        choices=['both', 'inter', 'intra'], help='Hemi links')
    parser.add_argument('--relation', default='intra', type=str,
                        choices=['both', 'inter', 'intra'],
                        help='ROI relation')
    parser.add_argument('--band', default='lga', type=str,
                        choices=__bands, help='Power band')
    parser.add_argument('--correction', default=1, type=int,
                        choices=[0, 1], help='Baseline correction')
    parser.add_argument('--biascorrection', default=0, type=int,
                        choices=[0, 1], help='Bias correction')
    parser.add_argument('--ds', default=0, type=int,
                        choices=[0, 2, 3, 4], help='Downsampling factor')
    args = parser.parse_args()
    model = str(args.model)
    metric = str(args.metric)
    hemi_links = str(args.hemilinks)
    relation = str(args.relation)
    band = str(args.band)
    correction = int(args.correction)
    biascorrection = int(args.biascorrection)
    ds = int(args.ds)

    # get metric settings
    if metric == 'ii':
        tail = 0  # ii = two tailed
    else:
        tail = 1  # others = one tailed

    # get mutual information type according to the model
    mi_type = get_mi_type(model)

    # set bias correction
    CONFIG["KW_GCMI"]["biascorrect"] = bool(biascorrection)

    # ================================= CONFIG ================================
    logger.info('=' * 79)
    logger.info(f"BAND={band}")
    logger.info(f"MODEL + MI_TYPE={model} + {mi_type}")
    logger.info(f"CONN METRIC={metric}")
    logger.info(f"BASELINE CORRECTION={bool(correction)}")
    logger.info(f"BIAS CORRECTION={bool(biascorrection)}")
    logger.info(f"DOWN-SAMPLING FACTOR={ds}")
    logger.info(f"INFERENCE={inference};MCP={mcp};TAIL={tail};NPERM={n_perm}")
    logger.info('=' * 79)

    # ================================== I/O ==================================
    # folder definition
    to_folder %= (band, model, metric)
    # savgol string
    savstr = 'none' if not isinstance(savgol, (int, float)) else str(savgol)
    # define how to save the file
    _save_as = (
        f"{metric}-{mi_type}-{band}_correction-{correction}_model-{model}_"
        f"bias-{biascorrection}_savgol-{savstr}_ds-{ds}_mcp-{mcp}_tail-{tail}_"
        f"inference-{inference}_nperm-{n_perm}_savgol-{savstr}_"
        f"{hemi_links}-hemi_relation-{relation}.nc"
    )
    # define the template for saving the file
    save_as = st.join(_save_as, folder=to_folder, force=True)

    # skip if already computed
    if os.path.isfile(save_as):
        logger.warning('------------ ALREADY COMPUTED. SKIPPED. ------------')
        exit()

    # ================================== PID ==================================
    ii, ii_p = [], []
    for n_s, s in enumerate(subjects):
        set_log_level(True)

        # -------------------------------- HGA --------------------------------
        # load the hga
        out = get_hga(
            s, band=band, savgol=savgol, ds=ds, is_conn=relation == 'inter',
            correction=correction, model=model
        )
        if out is None: continue

        # unwrap arguments
        _da, sfreq, hemi = out['data'], out['sfreq'], out['hemi']
        _da['trials'] = out['model']  # [::10]
        _da = _da.rename(trials='model')

        # ------------------------------ PID ----------------------------------
        # copnorm the data
        logger.info('    - Copnorm the data')
        _da.data = copnorm_nd(_da.data, axis=0)
        if mi_type == 'cc':
            _da['model'] = copnorm_nd(_da['model'].data, axis=0)

        _ii, _ii_p = compute_conn(
            _da, mi_type, metric, n_perm, -1, hemi, hemi_links, relation
        )
        ii.append(_ii)
        ii_p.append(_ii_p)

        del _da, _ii, _ii_p

    # =============================== RESHAPING ===============================
    # be sure that the time vector is the same across subjects
    times = ii[0]['times'].data
    for k in range(len(ii)):
        ii[k]['times'] = times
        ii_p[k]['times'] = times

    # get the unique list of pairs of roi
    roi_u = nonsorted_unique(np.concatenate([k['roi'].data for k in ii]))

    iir, iir_p, mi, roi = [], [], {}, []
    for r in roi_u:
        # find which subjects have this brain region
        _iir, _iir_p = [], []
        for n_s in range(len(ii)):
            is_roi = ii[n_s]['roi'].data == r
            if np.any(is_roi):
                _iir.append(ii[n_s].sel(roi=is_roi))
                _iir_p.append(ii_p[n_s].sel(roi=is_roi))

        # merge across contacts
        _iir = xr.concat(_iir, 'roi')
        _iir_p = xr.concat(_iir_p, 'roi')

        # track reshaping
        if len(_iir['roi']) > 1:
            iir.append(_iir)
            iir_p.append(_iir_p)
            mi[r] = _iir.mean('roi')
            roi.append(r)

        del _iir, _iir_p

    del ii, ii_p

    # ================================= STATS =================================
    set_log_level(True)
    dt = {}
    kw_stats = dict(inference=inference, mcp=mcp, tail=tail)

    _outs = compute_stat(
        '', [k.data for k in iir], [k.data for k in iir_p], times, roi,
        **kw_stats
    )
    dt.update(**_outs)

    xr.Dataset(dt).to_netcdf(save_as)

    st.runtime()
