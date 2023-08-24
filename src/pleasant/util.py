import numpy as np

__all__ = ['inv_variance_weighting', 'get_spectral_diffusion_rates']


def inv_variance_weighting(a):
    """
    Calculate inverse variance weighted average and standard deviation.
    :param a: two-column array containing values in the first and errors in the second column
    :return: average value, standard deviation
    """
    # filter out pairs where either value or error is nan
    a = a[~np.isnan(a).any(axis=1)]
    # filter out zero errors - would trigger zero division error
    a = a[a[:, 1] != 0]
    
    data = a[:, 0]
    weights = 1 / a[:, 1]**2
    avg, sum_of_weights = np.average(data, weights=weights, returned=True)
    std = np.sqrt(1 / sum_of_weights)
    return avg, std


def get_spectral_diffusion_rates(_df, verbose=False):
    """
    Compute the time-normalized spectral jumps of the fitted resonance frequency from one scan to the next.
    They make up the single data points of what is referred here as the spectral diffusion rate.
    Can only be determined if two consecutive scans have a fit. Back- and forward scans (retrace and trace) are
    treated separately, i.e. the jump from one forward (backward) to the next forward (backward) scan is computed.
    :param verbose: print statistics
    :param _df: data frame containing scan fit results in the format as retrieved from a Measurement object
    :return: two-column array of time-normalized spectral jumps (value and error)
    """
    rates = np.empty((0, 2))
    tot, with_fit, cons = 0, 0, 0

    # loop over measurements (uniquely identifiable by timestamp)
    for timestamp in _df['timestamp'].unique():
        # loop over trace and retrace
        for direction in [1, -1]:
            # select only the matching scans
            _df_sel = _df[(_df['timestamp'] == timestamp) & (_df['direction'] == direction)]

            # continue if there are no matching scans
            if _df_sel.empty:
                continue

            center = _df_sel['center'].to_numpy()
            err = _df_sel['center_stderr'].to_numpy()

            # determine jump_time: time between subsequent resonance acquisitions (in one direction, hence the factor 2)
            first_scan = _df_sel.iloc[0]
            scan_duration = first_scan['scan_range'] / first_scan['scan_speed']
            jump_time = 2 * (scan_duration + first_scan['break_duration'])

            # total number of scans
            tot += center.size
            # scans where a center was successfully fit
            with_fit += np.count_nonzero(~np.isnan(center))

            # spectral jump distance
            d = np.abs(center[1:] - center[:-1])
            # scans where a distance could be calculated - consecutively on
            cons += np.count_nonzero(~np.isnan(d))

            # calculating the error
            err_sq = err ** 2
            j_err = np.sqrt(err_sq[1:] + err_sq[:-1])

            combined = np.empty((d.size, 2))
            combined[:, 0] = d / jump_time
            combined[:, 1] = j_err / jump_time

            rates = np.append(rates, combined, axis=0)

    ratio_with = with_fit / tot
    ratio_cons = cons / tot
    if verbose:
        print(f'Total number of scans: {tot}')
        print(f'Scans with fit:        {with_fit} ({ratio_with:.0%})')
        print(f'Consecutive with fit:  {cons} ({ratio_cons:.0%})')
    return rates[~np.isnan(rates[:, 0])]
