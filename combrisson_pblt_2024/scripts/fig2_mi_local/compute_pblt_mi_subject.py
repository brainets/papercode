"""Compute conn (PID | II) + stats per condition in the time domain.

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

from frites.io import logger, set_log_level
from frites.config import CONFIG
from frites.workflow import WfMi, WfMiCombine
from frites.dataset import DatasetEphy

from pathta import Study

from research.study.pblt import (get_mi_type, get_hga)


###############################################################################
###############################################################################
#                             STATS FUNCTION
###############################################################################
###############################################################################


def compute_stat(name, ds, wf, kw_fit):
    """Compute the statistics.
    """
    out = {}
    out[f"mi_{name}"], out[f"pv_{name}"] = wf.fit(ds, **kw_fit)

    return out


###############################################################################
###############################################################################
#                                    MAIN
###############################################################################
###############################################################################


if __name__ == '__main__':
    ###########################################################################
    # filtering settings
    savgol = 1  # {float, None}

    # stat settings
    n_perm = 1000
    mcp = 'cluster'
    inference = 'ffx'

    # i/o settings
    to_folder = f'mi/subject/%s'
    ###########################################################################

    st = Study('PBLT')
    st.runtime()
    subjects = list(st.load_config('subjects.json').keys())

    # ================================= INPUTS ================================
    __bands = ['delta', 'theta', 'alpha', 'beta', 'lga', 'hga', 'bga', 'vlga']
    parser = argparse.ArgumentParser(description='Compute MI')
    parser.add_argument('--model', default='PE', type=str,
                        choices=['PE', 'outc', 'blocks', 'PC', 'surprise'],
                        help='Model-free of model-based analysis')
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
    band = str(args.band)
    correction = int(args.correction)
    biascorrection = int(args.biascorrection)
    ds = int(args.ds)

    # get mutual information type according to the model
    mi_type = get_mi_type(model)

    # set bias correction
    CONFIG["KW_GCMI"]["biascorrect"] = bool(biascorrection)

    # optional arguments
    kw_ds = dict(y='model', times='times', roi='roi')
    kw_wf = dict(inference=inference, mi_type=mi_type, verbose=False)
    kw_fit = dict(mcp=mcp, n_perm=n_perm, n_jobs=-1, random_state=0)

    # ================================== I/O ==================================
    # folder definition
    to_folder %= model
    # savgol string
    savstr = 'none' if not isinstance(savgol, (int, float)) else str(savgol)
    # define how to save the file
    _save_as = (
        f"mi-{mi_type}-{band}-{model}_correction-{correction}_"
        f"bias-{biascorrection}_ds-{ds}_mcp-{mcp}_inference-{inference}_"
        f"nperm-{n_perm}_savgol-{savstr}_s-%s.nc"
    )
    # define the template for saving the file
    save_as = st.join(_save_as, folder=to_folder, force=True)


    # ================================== PID ==================================
    for n_s, s in enumerate(subjects):
        # skip if already computed
        if os.path.isfile(save_as % s):
            logger.warning('---------- ALREADY COMPUTED. SKIPPED. ----------')
            continue

        x, dt = [], {}
        set_log_level(True)

        # ================================ HGA ================================
        # load the hga
        out = get_hga(
            s, band=band, savgol=savgol, ds=ds, is_conn=False,
            correction=correction, model=model
        )
        if out is None: continue

        # unwrap arguments
        _da, sfreq = out['data'], out['sfreq']
        cond, _model = out['cond'], out['model']  # [::10]

        # update roi dimension
        _da['roi'] = [f"{s}_{r}_{nr}" for nr, r in enumerate(_da['roi'].data)]

        # build multi-index
        _da['trials'] = pd.MultiIndex.from_arrays(
            (cond, _model), names=('cond', 'model')
        )

        x += [_da]
        # x += [_da.isel(roi=[r]) for r in range(len(_da['roi']))]

        # =============================== STATS ===============================

        # ________________________________ REW ________________________________
        set_log_level(True)
        logger.info("        Stats on REW")
        x_rew = [k.sel(cond='rew').copy() for k in x]
        ds_rew = DatasetEphy(x_rew, **kw_ds)
        wf_rew = WfMi(**kw_wf)
        _outs = compute_stat('rew', ds_rew, wf_rew, kw_fit)
        dt.update(**_outs)

        # ________________________________ PUN ________________________________
        set_log_level(True)
        logger.info("        Stats on PUN")
        x_pun = [k.sel(cond='pun').copy() for k in x]
        ds_pun = DatasetEphy(x_pun, **kw_ds)
        wf_pun = WfMi(**kw_wf)
        _outs = compute_stat('pun', ds_pun, wf_pun, kw_fit)
        dt.update(**_outs)

        xr.Dataset(dt).to_netcdf(save_as % s)

        del dt, x_rew, ds_rew, wf_rew, _outs, x_pun, ds_pun, wf_pun

    st.runtime()
