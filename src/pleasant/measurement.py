import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pleasant import fitting


class Measurement:
    def __init__(self, count_rate, exc_freq, timestamp=None, description=None, scan_duration=None, break_duration=None):
        """
        A Measurement represents a sequence of subsequent resonant absorption scans.
        The scanned excitation frequencies are assumed to be the same for each scan.
        Scans in different directions (sometimes referred to as trace and retrace) are supposed to be treated
        as individual measurements. Single scans are also supported.
        :param count_rate: 1D (only one scan) or 2D (multiple scans) numpy array containing the measured count rates
        :param exc_freq: 1D numpy array of the scanned excitation frequencies (in Hz)
        :param timestamp: string specifying when the measurement was taken
        :param description: details about the measurement, e.g. excitation power or source identifier
        :param scan_duration: time taken for an individual scan (in s)
        :param break_duration: time between individual scans (not accounting for potential backwards scan)
        """
        # make sure count_rate and exc_freq are arrays with the correct dimension
        # if count_rate is a 1D array (only one scan was performed for this measurement), convert to 2D
        if count_rate.ndim == 1:
            count_rate = count_rate.reshape(1, count_rate.size)
        try:
            if count_rate.ndim != 2:
                # it should be 2D by now
                raise AssertionError('count_rate should be a 1D or 2D numpy array')
        except AttributeError:
            raise AssertionError('count_rate should be a 1D or 2D numpy array')

        try:
            if exc_freq.ndim != 1:
                raise AssertionError('exc_freq should be a 1D numpy array')

        except AttributeError:
            raise AssertionError('exc_freq should be a 1D numpy array')

        if count_rate.shape[1] != exc_freq.size:
            raise AssertionError('count_rate and exc_freq should have the same bin count')

        # if all checks are passed, assign them to object attributes
        self._orig_count_rate = count_rate
        self._orig_exc_freq = exc_freq
        self.count_rate = self._orig_count_rate
        self.exc_freq = self._orig_exc_freq
        self._scan_direction = None

        self.timestamp = timestamp
        self.description = description
        self.scan_duration = scan_duration
        self.break_duration = break_duration

        self.sum_fit_result = None
        self.scan_fit_results = [None for _ in range(self.scan_count)]
        self.scan_fit_model = None
        self.photon_count_mask = np.full(self.scan_count, True)

    @property
    def scan_count(self):
        """Number of scans performed for this measurement."""
        return self.count_rate.shape[0]

    @property
    def bin_count(self):
        """Number of bins a scan is divided into."""
        return self.count_rate.shape[1]

    @property
    def freq_range(self):
        """Scanned frequency range in Hz."""
        return self.exc_freq.max() - self.exc_freq.min()

    @property
    def scan_direction(self):
        """Returns +1 or -1 depending on whether the scan is performed
        towards positive or negative frequency direction."""
        if self._scan_direction:
            return self._scan_direction
        else:
            d = np.diff(self.exc_freq)
            if np.all(d > 1):
                self._scan_direction = 1
            elif np.all(d < 1):
                self._scan_direction = -1
            else:
                raise AssertionError('exc_freq should be monotonically in- or decreasing')
            return self._scan_direction

    @property
    def scan_speed(self):
        """Scan speed in Hz/s."""
        return self.freq_range / self.scan_duration

    @property
    def bin_width(self):
        """Frequency range that a single bin spans (in Hz)."""
        return self.freq_range / self.bin_count

    def print_info(self):
        """Print information on this measurement."""
        print('Measurement')
        print(f'{self.timestamp} | {self.description} | {self.scan_direction:+} direction')
        print(f'Scan over {1e-9 * self.freq_range :.2f} GHz at {1e-9 * self.scan_speed:.2f} GHz/s')

    def rebin_data(self, bins_to_merge=None, target_bin_width=None):
        """
        Rebin the count rate matrix and the frequency vector to a lower resolution than the original,
        increasing the bin width. You can specify either a number of bins to merge or a target bin width.
        If necessary, bins at the high frequency end will be trimmed.
        All previously performed fits and masks will be deleted.
        :param bins_to_merge: Number of bins to merge and average over. Factor that the bin count is reduced by.
        :param target_bin_width: Target bin width in Hz. A number of bins to merge will be calculated from this value.
        """
        if not (bins_to_merge or target_bin_width):
            raise AssertionError('Either size or bin_width must be specified.')
        elif bins_to_merge and target_bin_width:
            raise AssertionError('Either size or bin_width must be specified, not both.')

        # reset frequency and count_rate matrix to original binning
        self.count_rate = self._orig_count_rate
        self.exc_freq = self._orig_exc_freq

        # reset potentially existing fit results and mask
        self.sum_fit_result = None
        self.scan_fit_results = [None for _ in range(self.scan_count)]
        self.scan_fit_model = None
        self.photon_count_mask = np.full(self.scan_count, True)

        orig_bin_width = self.bin_width

        # if bin_width is given, calculate number of bins to merge
        if target_bin_width:
            bins_to_merge = max(1, round(target_bin_width / orig_bin_width))

        # trim some bins such that the original bin count can be divided by bins_to_merge
        remainder = self.count_rate.shape[1] % bins_to_merge
        if remainder != 0:
            self.count_rate = self.count_rate[:, :-remainder]
            self.exc_freq = self.exc_freq[:-remainder]

        # do the actual rebinning: since we are working with rates, taking the mean is appropriate
        new_bin_count = self.count_rate.shape[1] // bins_to_merge
        self.count_rate = self.count_rate.reshape(self.count_rate.shape[0], new_bin_count, bins_to_merge).mean(axis=2)
        self.exc_freq = self.exc_freq.reshape(-1, bins_to_merge).mean(axis=1)

        # report on result
        new_bin_width = self.bin_width
        print((f'Rebinned from {1e-6*orig_bin_width:.1f} to {1e-6*new_bin_width:.1f} MHz/bin,'
               f' trimming {remainder} bin(s).'))

    def plot_sum_of_scans(self, x_lim=None):
        """
        Plot the count rates as a 2D image, sum up the counts of all scans and fit them with a Gaussian function.
        :param x_lim: limits for the x-axis
        :return: matplotlib figure object
        """
        if self.scan_count < 2:
            raise AssertionError('Measurement must contain at least two scans.')

        if self.sum_fit_result is None:
            self.fit_sum_of_scans()

        fig, (ax, ax2) = plt.subplots(2, sharex=True, constrained_layout=True)

        # top plot: inhomogeneous line width (sum of scans)
        ax.plot(1e-9 * self.exc_freq, self.sum_fit_result.data, label='Data')
        ax.plot(1e-9 * self.exc_freq, self.sum_fit_result.best_fit, label='Gaussian Fit')

        fwhm_ghz = 1e-9 * self.sum_fit_result.params['fwhm'].value
        if fwhm_ghz < 1.0:
            label = '$w$ = {:.0f} MHz'.format(1e3 * fwhm_ghz)
        else:
            label = '$w$ = {:.2f} GHz'.format(fwhm_ghz)
        ax.plot([], [], 'w', label=label)

        ax.ticklabel_format(scilimits=(-5, 3))
        ax.set_ylabel('Counts per Bin')

        # bottom plot: evolution of spectral position
        scan_index = np.arange(self.scan_count)
        cf = ax2.contourf(1e-9 * self.exc_freq, scan_index, 1e-3 * self.count_rate)

        # colorbar
        # first plot has no colorbar, create spacer
        divider1 = make_axes_locatable(ax)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cax1.axis('off')

        # for actual colorbar
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(cf, ax=ax2, cax=cax2)
        cbar2.set_label('Count Rate (kHz)')

        ax2.set_xlabel('Excitation Frequency (GHz)')
        ax2.set_ylabel('Scan Index')

        ax.set_title(f'{self.timestamp} | {self.description}')
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_ylim(0)
        ax.legend()

        if x_lim is not None:
            ax2.set_xlim(x_lim)

        return fig

    def fit_sum_of_scans(self):
        """
        Sum up the counts of all scans and fit them with a Gaussian function.
        Save the result internally to self.sum_fit_result
        :return:
        """
        time_per_bin = self.bin_width / self.scan_speed
        counts = self.count_rate * time_per_bin
        sum_of_scans = counts.sum(axis=0)

        model = fitting.gaussian
        params = model.make_params()
        sigma_guess = fitting.gauss_sigma(self.freq_range / 2)
        sigma_max = fitting.gauss_sigma(self.freq_range)
        amp_guess = fitting.gauss_amplitude(sum_of_scans.max(), sigma_guess)
        center_guess = self.exc_freq[sum_of_scans.argmax()]

        params['amplitude'].set(value=amp_guess, min=0)
        params['sigma'].set(value=sigma_guess, max=sigma_max)
        params['center'].set(value=center_guess)

        self.sum_fit_result = model.fit(sum_of_scans, x=self.exc_freq, params=params)

    def photon_count_filter(self, threshold):
        """
        Creates a mask depending on a very simple photon count filtering condition.
        The mask is used when scans are fitted later.
        If a scan contains a single bin in which at least as many counts were registered as the threshold,
        it passes the filter.
        :param threshold: minimum counts in a bin to pass the filter
        :return: mask array
        """
        # reset potentially existing fit results
        self.scan_fit_results = [None for _ in range(self.scan_count)]
        self.scan_fit_model = None

        # convert count rate to counts, round up
        bin_time = self.scan_duration / self.bin_count
        absolute_photon_count = (self.count_rate * bin_time).round(0).astype(int)

        # select scans that contain at least one bin with at least as many photons as the threshold
        self.photon_count_mask = (absolute_photon_count >= threshold).any(axis=1)
        return self.photon_count_mask

    def fit_scans(self, model_name='Lorentzian', fwhm_guess=50e6):
        """
        Fit all scans with a peak-like model.
        :param model_name: name of the model to use for fitting, can be Lorentzian, Gaussian, Pseudo Voigt and Voigt.
        :param fwhm_guess: initial value to use for the FWHM
        :return:
        """
        self.scan_fit_model = model_name
        params = None

        offset = self.exc_freq.mean()
        exc_freq_centered_ghz = 1e-9 * (self.exc_freq - offset)
        fwhm_guess_ghz = 1e-9 * fwhm_guess
        fwhm_max_ghz = exc_freq_centered_ghz.max() - exc_freq_centered_ghz.min()

        if model_name == 'Gaussian':
            model = fitting.gaussian
            sigma_guess = fitting.gauss_sigma(fwhm_guess_ghz)
            sigma_max = fitting.gauss_sigma(fwhm_max_ghz)
        elif model_name == 'Lorentzian':
            model = fitting.lorentzian
            sigma_guess = fitting.lorentz_sigma(fwhm_guess_ghz)
            sigma_max = fitting.lorentz_sigma(fwhm_max_ghz)
        elif model_name == 'Pseudo Voigt':
            model = fitting.pseudo_voigt
            sigma_guess = fitting.lorentz_sigma(fwhm_guess_ghz)
            sigma_max = fitting.lorentz_sigma(fwhm_max_ghz)
        elif model_name == 'Voigt':
            model = fitting.voigt
            sigma_guess = fitting.voigt_sigma(fwhm_guess_ghz)
            sigma_max = fitting.voigt_sigma(fwhm_max_ghz)

            params = model.make_params()
            params['gamma'].set(value=sigma_guess, vary=True, expr=None, min=0)
        else:
            raise AssertionError('Unknown fitting model.')

        if params is None:
            params = model.make_params()

        params['sigma'].set(value=sigma_guess, max=sigma_max)
        params['amplitude'].set(value=8000)

        results = []
        for rate_single_scan, enough_photons in zip(self.count_rate, self.photon_count_mask):
            if not enough_photons:
                results.append(None)
                continue
            f_max = exc_freq_centered_ghz[rate_single_scan.argmax()]
            params['center'].set(value=f_max)
            try:
                fit_result = model.fit(rate_single_scan, x=exc_freq_centered_ghz, params=params)
            # unsuccessful fit
            except ValueError:
                results.append(None)
            else:
                results.append(fit_result)

        self.scan_fit_results = results

    def plot_individual_scan(self, i, freq_range=None, fit_eval_density=1):
        """
        Plot an individual scan. The fit is included if it was performed before.
        :param i: Scan index to plot.
        :param freq_range: If specified, trim the plot to this range (in Hz) around the fitted center frequency.
        :param fit_eval_density: factor to increase fit smoothness by evaluation at more data points
        :return: matplotlib figure object
        """

        # generate a meaningful title
        speed = 1e-9 * self.scan_speed
        n = self.scan_count
        title = f'{self.timestamp} | {self.description} | {self.scan_direction:+} | {speed:.1f} GHz/s | {i}/{n}'

        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_title(title)

        ax.plot(1e-9 * self.exc_freq, 1e-3 * self.count_rate[i], '.-', label='Data')

        # if there is a fit result, plot the fit and some info
        fit_result = self.scan_fit_results[i]
        if fit_result:
            # fitting is done with a recentered and rescaled exc_freq
            offset = self.exc_freq.mean()
            exc_freq_centered_ghz = 1e-9 * (self.exc_freq - offset)
            freq_fit = np.linspace(exc_freq_centered_ghz.min(), exc_freq_centered_ghz.max(),
                                   fit_eval_density * exc_freq_centered_ghz.size)
            rate_fit = fit_result.eval(x=freq_fit)
            freq_max = 1e-9 * offset + freq_fit[rate_fit.argmax()]

            # reset centering and rescaling
            fwhm = 1e9 * fit_result.params['fwhm'].value
            center = offset + 1e9 * fit_result.params['center'].value

            ax.plot(1e-9 * offset + freq_fit, 1e-3 * rate_fit, 'r', label=self.scan_fit_model + ' Fit')
            ax.plot([], [], 'w', label=f'w = {1e-6 * fwhm:.0f} MHz')
            ax.plot([], [], 'w', label=f'$f_c$ = {1e-12 * center:.5f} THz')

        # set limits if specified
        if fit_result and freq_range:
            freq_range_ghz = 1e-9 * freq_range
            ax.set_xlim([freq_max - freq_range_ghz / 2, freq_max + freq_range_ghz / 2])
        else:
            ax.autoscale(axis='x', tight=True)

        ax.set_ylim(0)
        ax.set_xlabel('Laser Frequency (GHz)')
        ax.set_ylabel('APD Signal (kc/s)')
        ax.legend(loc='upper right')

        return fig

    @property
    def scan_fit_data(self):
        """Scan fit data as a pandas data frame. Rows correspond to individual scans."""
        # check if there are any successful fits
        if self.scan_fit_results.count(None) == len(self.scan_fit_results):
            df_fit_results = pd.DataFrame()
        # there is at least one successful fit
        else:
            list_of_dicts = []
            params = ['fwhm', 'height', 'center', 'c']
            nan_dict = {param: np.nan for param in params} | {f'{param}_stderr': np.nan for param in params}
            offset = self.exc_freq.mean()

            # collect the fit results
            for res in self.scan_fit_results:
                if res:
                    value_dict = {param: res.params[param].value for param in params}
                    stderr_dict = {f'{param}_stderr': res.params[param].stderr for param in params}

                    # replace None with nan for stderr
                    for key in stderr_dict:
                        if stderr_dict[key] is None:
                            stderr_dict[key] = np.nan

                    # reset centering and rescaling
                    value_dict['fwhm'] *= 1e9
                    value_dict['center'] = offset + 1e9 * value_dict['center']
                    stderr_dict['fwhm_stderr'] *= 1e9
                    stderr_dict['center_stderr'] *= 1e9

                    # determine max count rate within fitted fwhm
                    value_dict['max_rate_in_fwhm'] = fitting.max_within_fwhm(
                        self.exc_freq, res.data, value_dict['center'], value_dict['fwhm']
                    )

                    list_of_dicts.append(value_dict | stderr_dict)
                else:
                    # if res is None (unsuccessful fit), put nan for every value
                    list_of_dicts.append(nan_dict)

            df_fit_results = pd.DataFrame(list_of_dicts)

        columns = ['timestamp', 'description', 'direction', 'scan_speed', 'scan_range', 'break_duration', 'bin_width',
                   'fit_model']
        df_meas_info = pd.DataFrame(columns=columns)

        df_index = pd.DataFrame(np.arange(self.scan_count), columns=['scan_index'])

        # concatenate the columns together
        df = pd.concat([df_meas_info, df_index, df_fit_results], axis=1)

        df['timestamp'] = self.timestamp
        df['description'] = self.description
        df['direction'] = self.scan_direction
        df['scan_speed'] = self.scan_speed
        df['scan_range'] = self.freq_range
        df['break_duration'] = self.break_duration
        df['bin_width'] = self.bin_width
        df['fit_model'] = self.scan_fit_model

        return df
