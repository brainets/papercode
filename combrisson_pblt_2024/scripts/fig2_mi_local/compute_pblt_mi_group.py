import os

import numpy as np
import xarray as xr
import pandas as pd
import argparse

from frites.io import logger, set_log_level
from frites.config import CONFIG
from frites.workflow import WfMi, WfMiCombine, WfStats
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
    __dt = {}
    for sigma in [0., 0.001, 0.01, 0.1]:
        # fit the workflow
        mi, pv = wf.fit(ds, rfx_sigma=sigma, **kw_fit)
        tv = wf.get_params('tvalues')
        attrs = wf.attrs.copy()

        # fill outputs
        sstr = "%.3f" % sigma
        __dt[f"mi_{name}"] = mi
        __dt[f"pv_{name}_{sstr[2:]}"] = pv
        __dt[f"tv_{name}_{sstr[2:]}"] = tv

    # conjunction analysis
    roi_c, times = wf._roi, mi['times'].data
    conj, conj_roi = [], []
    for n_r, r in enumerate(roi_c):
        for n_c in range(wf.mi[n_r].shape[0]):
            # fit at the single contact level
            _conj = WfStats(verbose=False).fit(
                [wf.mi[n_r][[n_c], :]], [wf.mi_p[n_r][:, [n_c], :]],
                inference='ffx')[0]

            # xarray transformation
            _conj = xr.DataArray(_conj.squeeze(), dims=('times',),
                                 coords=(times,))
            conj.append(_conj)
            conj_roi.append(r)
    conj = xr.concat(conj, 'roi').T
    conj['roi'] = conj_roi

    # confidence interval
    __dt[f"ci_{name}"] = wf.get_params(
        'mi_ci', cis=[90, 95, 99, 'sd', 'sem'], n_boots=kw_fit['n_perm'],
        random_state=0
    )

    return __dt, conj


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
    to_folder = f'mi/group/%s'
    ###########################################################################

    st = Study('PBLT')
    st.runtime()
    subjects = list(st.load_config('subjects.json').keys())  # [0:2]

    # ================================= INPUTS ================================
    __bands = ['delta', 'theta', 'alpha', 'beta', 'lga', 'hga', 'bga', 'vlga']
    parser = argparse.ArgumentParser(description='Compute MI')
    parser.add_argument('--model', default='PE', type=str,
                        choices=['PE', 'outc', 'blocks', 'PC', 'surprise',
                                 'PE_outc'],
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
    if model in ['PE_outc']:
        mi_type = 'ccd'
    else:
        mi_type = get_mi_type(model)

    # set bias correction
    CONFIG["KW_GCMI"]["biascorrect"] = bool(biascorrection)

    # ================================== I/O ==================================
    # folder definition
    to_folder %= model

    # savgol string
    savstr = 'none' if not isinstance(savgol, (int, float)) else str(savgol)

    # define how to save the file
    _save_as = (
        f"mi-{mi_type}-{band}-{model}_correction-{correction}_"
        f"bias-{biascorrection}_ds-{ds}_mcp-{mcp}_inference-{inference}_"
        f"nperm-{n_perm}_savgol-{savstr}.nc"
    )

    # define the template for saving the file
    save_as = st.join(_save_as, folder=to_folder, force=True)

    # for ccd, keep only model name
    if mi_type == 'ccd':
        model = model.split('_')[0]

    # skip if already computed
    if os.path.isfile(save_as):
        logger.warning('------------ ALREADY COMPUTED. SKIPPED. ------------')
        exit()

    # ================================== PID ==================================
    x = []
    for n_s, s in enumerate(subjects):
        set_log_level(True)

        # -------------------------------- HGA --------------------------------
        # load the hga
        out = get_hga(
            s, band=band, savgol=savgol, ds=ds, is_conn=False,
            correction=correction, model=model
        )
        if out is None: continue

        # unwrap arguments
        _da, sfreq = out['data'], out['sfreq']
        cond, outc, _model = out['cond'], out['outc'], out['model']  # [::10]

        # build multi-index
        _da['trials'] = pd.MultiIndex.from_arrays(
            (cond, outc, _model), names=('cond', 'outc', 'model')
        )

        x += [_da.isel(roi=[r]) for r in range(len(_da['roi']))]

    # ================================= STATS =================================
    dt = {}
    kw_ds = dict(y='model', times='times', roi='roi', nb_min_suj=2)
    kw_wf = dict(inference=inference, mi_type=mi_type, verbose=False)
    kw_fit = dict(mcp=mcp, n_perm=n_perm, n_jobs=-1, random_state=0)
    if mi_type == 'ccd':
        kw_ds['z'] = 'outc'

    # __________________________________ REW __________________________________
    set_log_level(True)
    logger.info("        Stats on REW")
    x_rew = [k.sel(cond='rew').copy() for k in x]
    ds_rew = DatasetEphy(x_rew, **kw_ds)
    wf_rew = WfMi(**kw_wf)
    _outs, conj_rew = compute_stat('rew', ds_rew, wf_rew, kw_fit)
    dt.update(**_outs)

    # __________________________________ PUN __________________________________
    set_log_level(True)
    logger.info("        Stats on PUN")
    x_pun = [k.sel(cond='pun').copy() for k in x]
    ds_pun = DatasetEphy(x_pun, **kw_ds)
    wf_pun = WfMi(**kw_wf)
    _outs, conj_pun = compute_stat('pun', ds_pun, wf_pun, kw_fit)
    dt.update(**_outs)

    # save main results
    xr.Dataset(dt).to_netcdf(save_as)

    # save conjunction
    xr.Dataset({
        'rew': conj_rew,
        'pun': conj_pun
    }).to_netcdf(save_as.replace('.nc', '-conj.nc'))

    st.runtime()
