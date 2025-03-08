"""Module containing Classes providing a simple interface to UniDec and ThermoRawFileReader
Author: Simon Knoblauch
Email: simonb.knoblauch@gmail.com
"""

import json
import os
from datetime import datetime
from glob import glob
from itertools import combinations, product

import clr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unidec

clr.AddReference("Lib/ThermoFisher.CommonCore.Data")
clr.AddReference("Lib/ThermoFisher.CommonCore.RawFileReader")
clr.AddReference("Lib/ThermoFisher.CommonCore.BackgroundSubtraction")
clr.AddReference("Lib/ThermoFisher.CommonCore.MassPrecisionEstimator")

import os

from ThermoFisher.CommonCore.Data import Extensions, ToleranceUnits
from ThermoFisher.CommonCore.Data.Business import (
    ChromatogramSignal,
    ChromatogramTraceSettings,
    DataUnits,
    Device,
    GenericDataTypes,
    SampleType,
    Scan,
    TraceType,
)
from ThermoFisher.CommonCore.Data.FilterEnums import IonizationModeType, MSOrderType
from ThermoFisher.CommonCore.Data.Interfaces import (
    IChromatogramSettings,
    IScanEventBase,
    IScanFilter,
    RawFileClassification,
)
from ThermoFisher.CommonCore.MassPrecisionEstimator import PrecisionEstimate
from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter


def make_output_folder(
    data_in: str, folder_name: str, base_folder=r"../../../Analyses"
):
    relpath_data = os.path.relpath(data_in, os.getcwd())
    data_type = relpath_data.split("\\")[4]
    folder_out = (
        f"{base_folder}/{data_type}/{datetime.today().strftime('%Y%m%d')}/{folder_name}"
    )
    os.makedirs(folder_out, exist_ok=True)

    return folder_out


class SpectraExtractor:
    def __init__(self, filepath) -> None:
        self.file = filepath
        self.name = os.path.basename(filepath).split(".")[0]
        self.raw_file = None
        self.chrom = None
        self.range = None
        self.spectrum = None

    def get_chromatogram(self):
        raw_file = RawFileReaderAdapter.FileFactory(self.file)
        raw_file.SelectInstrument(Device.MS, 1)

        first_scan = raw_file.RunHeaderEx.FirstSpectrum
        last_scan = raw_file.RunHeaderEx.LastSpectrum
        settings = ChromatogramTraceSettings(TraceType.TIC)

        data = raw_file.GetChromatogramData([settings], first_scan, last_scan)
        trace = ChromatogramSignal.FromChromatogramData(data)

        self.chrom = np.array(
            (
                np.arange(first_scan, last_scan + 1),
                list(trace[0].Times),
                list(trace[0].Intensities),
            )
        )
        raw_file.Dispose()

        return self.chrom

    def plot_chromatogram(self, plot_time: bool = False, ax=None):
        if self.chrom is None:
            self.get_chromatogram()

        if ax is None:
            fig, ax = plt.subplots()

        if plot_time:
            x_axis = self.chrom[1]
            xlabel = "RT [min]"
        else:
            x_axis = self.chrom[0]
            xlabel = "Scan Nr."

        ax.plot(x_axis, self.chrom[2], linewidth=1, color="indigo")
        ax.set(ylabel="TIC [a.u.]", xlabel=xlabel)
        ax.locator_params(axis="x", nbins=20)

        if self.range is not None:
            ax.hlines(
                0, x_axis[self.range[0] - 1], x_axis[self.range[1] - 1], colors="k"
            )

    def get_average_spec(self, range=None, tolerance_ppm=5.0):
        raw_file = RawFileReaderAdapter.FileFactory(self.file)
        raw_file.SelectInstrument(Device.MS, 1)

        if range is None:
            range = [
                raw_file.RunHeaderEx.FirstSpectrum,
                raw_file.RunHeaderEx.LastSpectrum,
            ]

        self.range = range
        scan_stats = raw_file.GetScanStatsForScanNumber(range[0])
        scan_filter = IScanFilter(raw_file.GetFilterForScanNumber(range[0]))

        options = Extensions.DefaultMassOptions(raw_file)
        options.ToleranceUnits = ToleranceUnits.ppm
        options.Tolerance = tolerance_ppm
        average_scan = Extensions.AverageScansInScanRange(
            raw_file, *range, scan_filter, options
        )

        if scan_stats.IsCentroidScan:
            centroid_stream = average_scan.CentroidScan

            self.spectrum = np.array(
                (list(centroid_stream.Masses), list(centroid_stream.Intensities))
            ).T

        else:
            segmented_scan = average_scan.SegmentedScan
            self.spectrum = np.array(
                (list(segmented_scan.Positions), list(segmented_scan.Intensities))
            )

        raw_file.Dispose()

        return self.spectrum

    def get_spectrum(self, scan_no):
        raw_file = RawFileReaderAdapter.FileFactory(self.file)
        raw_file.SelectInstrument(Device.MS, 1)

        scan_stats = raw_file.GetScanStatsForScanNumber(scan_no)

        if scan_stats.IsCentroidScan:
            centroid_stream = raw_file.GetCentroidStream(scan_no, False)

            self.spectrum = np.array(
                (list(centroid_stream.Masses), list(centroid_stream.Intensities))
            )

        else:
            segmented_scan = raw_file.GetSegmentedScanFromScanNumber(
                scan_no, scan_stats
            )

            self.spectrum = np.array(
                (list(segmented_scan.Positions), list(segmented_scan.Intensities))
            )

        raw_file.Dispose()

        return self.spectrum

    def plot_spectrum(self, zoom=None, ax=None):
        assert (
            self.spectrum is not None
        ), "Please select a Spectrum first using :get_average_spec: or :get_spectrum:!"

        if ax is None:
            fig, ax = plt.subplots()

        if zoom is not None:
            idx = (self.spectrum[0] > [zoom[0]]) & (self.spectrum[0] < [zoom[1]])
            x = self.spectrum[0, idx]
            y = self.spectrum[1, idx] / self.spectrum[1, idx].max()
        else:
            x = self.spectrum[0]
            y = self.spectrum[1] / self.spectrum[1].max()

        ax.plot(x, y, linewidth=1, color="indigo")
        ax.set(xlabel="m/z [Th]", ylabel="Rel. Int [a.u]")
        ax.locator_params(axis="x", nbins=30)

    def save_spectrum(self, folder):
        assert (
            self.spectrum is not None
        ), "Please select a Spectrum first using :get_average_spec: or :get_spectrum:!"

        if self.range is not None:
            add_range = f"_range_{self.range[0]}-{self.range[1]}"
        else:
            add_range = ""

        np.savetxt(f"{folder}\\{self.name}_selected_{add_range}.txt", self.spectrum.T)


class UnidecInterface(unidec.UniDec):
    """Simple Interface to use the UniDec engine for pyhton scripts"""

    def __init__(self, data_in: str, folder_out: str, *args, **kwargs):
        unidec.UniDecEngine.__init__(self)
        data_name = os.path.basename(data_in).split(".")[0]
        self.data_input = data_in
        self.folder_out = folder_out

        self.config.outfname = f"{self.folder_out}/{data_name}_unidec"
        self.outfile = self.config.outfname
        self.results = {}

    def run(self, config_input: str = None):
        self.open_file(self.data_input)
        if config_input:
            self.load_config(config_input)
        self.autorun()

    def check_folder(self):
        if not os.path.exists(self.folder_out):
            os.mkdir(self.folder_out)

    def save_config_json(self, filename):
        config = self.config.get_config_dict()

        with open(filename, "w") as output:
            json.dump(config, output, indent=4)

    def load_results(self):
        for file in glob(f"{self.folder_out}\\*_unidec*"):
            match file.split("_")[-1]:
                case "input.dat":
                    self.results["mz_data"] = np.genfromtxt(file).T
                case "mass.txt":
                    self.results["mass_data"] = np.genfromtxt(file).T
                case "peaks.dat":
                    array = np.genfromtxt(file)
                    if array.ndim == 1:
                        array = array.reshape(1, array.shape[0])
                    self.results["mass_peaks"] = array
                case "mzpeakdata.dat":
                    array = np.genfromtxt(file)
                    if array.ndim == 1:
                        array = array.reshape(1, array.shape[0])
                    self.results["mz_peaks"] = array
                case "peakparam.dat":
                    self.results["peak_table"] = pd.read_csv(
                        file,
                        delimiter=" ",
                        header=None,
                        skiprows=1,
                        names=[
                            "Mass",
                            "MassStdGuess",
                            "AvgCharge",
                            "StdDevCharge",
                            "Height",
                            "Area",
                            "MassCentroid",
                            "MassFWHM",
                            "MassErrorBetweenZ",
                            "DScore",
                        ],
                    )
        self.results["markers"] = ["o", "s", "p", "D", "<", ">", "^", "v"]

        cmap = matplotlib.colormaps["rainbow"]
        self.results["colorvals"] = cmap(
            np.linspace(0, 1, self.results["mass_peaks"].shape[0])
        )

    def plot_mass_data(
        self,
        zoom=None,
        ax=None,
        display_masses=False,
    ):
        if ax is None:
            fig, ax = plt.subplots()

        if zoom is not None:
            ax.set(xlim=zoom)

        ax.plot(
            self.results["mass_data"][0],
            100 * self.results["mass_data"][1] / self.results["mass_data"][1].max(),
            zorder=1,
            color="darkblue",
            linewidth=1,
        )
        if len(self.results["mass_peaks"].shape) > 1:
            for i, peak in enumerate(self.results["mass_peaks"]):
                ax.scatter(
                    peak[0],
                    peak[1],
                    marker=self.results["markers"][i % 8],
                    color=self.results["colorvals"][i],
                    zorder=10,
                    edgecolor="k",
                )

                if display_masses is True and zoom is not None:
                    if peak[0] > zoom[0] and peak[0] < zoom[1]:
                        ax.text(peak[0], peak[1] + 2, f"{peak[0]} Da", fontsize=10)
                elif display_masses is True:
                    ax.text(peak[0], peak[1] + 2, f"{peak[0]} Da", fontsize=10)

                self.results["peak_table"].loc[i, "Marker"] = self.results["markers"][
                    i % 8
                ]
        ax.set(xlabel="Mass [Da]", ylabel="Rel. Int [%]")
        ax.locator_params(axis="x", nbins=20)

    def plot_mz_data(self, zoom=None, ax=None, threshold=0.1, display_charges=False):
        if ax is None:
            fig, ax = plt.subplots()

        if zoom is not None:
            ax.set(xlim=zoom)
        else:
            ax.set(
                xlim=[
                    self.results["mz_data"][0].min() - 500,
                    self.results["mz_data"][0].max() + 500,
                ]
            )

        ax.plot(
            self.results["mz_data"][0],
            self.results["mz_data"][1],
            color="darkblue",
            linewidth=1,
            zorder=1,
        )
        if len(self.results["mz_peaks"].shape) > 1:
            for i, peaks in enumerate(self.results["mz_peaks"][:]):
                yvals = np.interp(
                    peaks, self.results["mz_data"][0], self.results["mz_data"][1]
                )
                ii = yvals > threshold
                pks = peaks[ii]
                y = yvals[ii]

                ax.scatter(
                    pks,
                    y,
                    marker=self.results["markers"][i % 8],
                    color=self.results["colorvals"][i],
                    zorder=5,
                    edgecolor="k",
                )

                if display_charges is True:
                    charges = np.rint(self.results["mass_peaks"][i][0] / pks).astype(
                        "int"
                    )
                    for iii, peak in enumerate(pks):
                        if zoom is not None:
                            if peak > zoom[0] and peak < zoom[1]:
                                ax.text(
                                    peak,
                                    y[iii] + 0.02,
                                    charges[iii],
                                    fontsize=10,
                                    zorder=10,
                                )
                        else:
                            ax.text(
                                peak,
                                y[iii] + 0.02,
                                charges[iii],
                                fontsize=10,
                                zorder=10,
                            )

        ax.set(xlabel="m/z [Th]", ylabel="Rel. Int [a.u]")
        ax.locator_params(axis="x", nbins=20)

    def get_peak_df(self):
        return self.results["peak_table"]

    def plot_table(self, ax):
        df = self.get_peak_df()[["Marker", "Mass", "MassStdGuess", "Area"]]
        peak_nr = df.index.size
        size_bbox = 0.05 * peak_nr + 0.05

        ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="top",
            colWidths=[0.05, 0.1, 0.1, 0.1],
            rowColours=self.results["colorvals"],
            bbox=[0.75, 1 - size_bbox, 0.25, size_bbox],
        )


def get_likely_composition(
    array, mass, no_subunits: int, max_multiplicates: int = 1, tolerance=10
):
    df = pd.DataFrame(
        product(range(0, max_multiplicates + 1), repeat=len(array)), columns=array
    )
    df = df[df.sum(1) <= no_subunits]
    df["sum"] = df[array] @ array
    df["diff"] = np.abs(df["sum"] - mass)
    df = df[df["diff"] <= tolerance].copy()
    filtered = df.sort_values("diff")
    return filtered


def get_all_differences(df):
    combis = list(combinations(df.index, 2))
    diff = [np.abs(df.loc[m1, "Mass"] - df.loc[m2, "Mass"]) for m1, m2 in combis]
    idxs = [f"{m1}, {m2}" for m1, m2 in combis]
    markers = [f"{df.loc[m1,'Marker']} - {df.loc[m2,'Marker']}" for m1, m2 in combis]

    return pd.DataFrame(
        {
            "Indexes": idxs,
            "Markers": markers,
            "Difference": diff,
        }
    )


def get_neighbour_diffs(df, window=2):
    rg = df.index
    df_list = []
    for i in rg:
        idx = range(max(0, i - window), min(rg.size - 1, i + window) + 1)
        diffs = [np.abs(df.loc[ii, "Mass"] - df.loc[i, "Mass"]) for ii in idx]
        markers = df.loc[idx, "Marker"]

        df_list.append(
            pd.DataFrame(
                {
                    "Ref_Peak_idx": i,
                    "Ref_Peak_Marker": df.loc[i, "Marker"],
                    "Peak_idx": idx,
                    "Peak_Marker": markers,
                    "Difference": diffs,
                }
            )
        )

    return pd.concat(df_list, ignore_index=True)


def get_stoichiometries(df, mass_array, no_subunits, max_multiplicates):
    comp_list = []
    for i in df.index:
        comp = get_likely_composition(
            array=mass_array,
            mass=df.loc[i, "Mass"],
            no_subunits=no_subunits,
            max_multiplicates=max_multiplicates,
            tolerance=np.inf,
        ).iloc[0]
        comp["Mass_i"] = i
        comp["Mass"] = df.loc[i, "Mass"]
        comp_list.append(comp)

    return pd.concat(comp_list, axis=1, ignore_index=True).T
