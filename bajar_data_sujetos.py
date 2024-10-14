# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import mne
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import scipy.signal as sc

import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

#%% 

def eda_methods_nuevo(
    sampling_rate=1000,
    method="default",
    method_cleaning="default",
    method_peaks="default",
    method_phasic="cvxeda",
    **kwargs,
):
    """
    Modificación para que el método de EDA_Tonic sea cvxOPT
    Cambio el default de method_phasic a cvxeda (ahí arriba)
    """
    
    # Sanitize inputs
    method_cleaning = str(method).lower() if method_cleaning == "default" else str(method_cleaning).lower()
    method_phasic = str(method).lower() if method_phasic == "default" else str(method_phasic).lower()
    method_peaks = str(method).lower() if method_peaks == "default" else str(method_peaks).lower()

    # Create dictionary with all inputs
    report_info = {
        "sampling_rate": sampling_rate,
        "method_cleaning": method_cleaning,
        "method_phasic": method_phasic,
        "method_peaks": method_peaks,
        "kwargs": kwargs,
    }

    # Get arguments to be passed to underlying functions
    kwargs_cleaning, report_info = nk.misc.report.get_kwargs(report_info, nk.eda_clean)
    kwargs_phasic, report_info = nk.misc.report.get_kwargs(report_info, nk.eda_phasic)
    kwargs_peaks, report_info = nk.misc.report.get_kwargs(report_info, nk.eda_peaks)

    # Save keyword arguments in dictionary
    report_info["kwargs_cleaning"] = kwargs_cleaning
    report_info["kwargs_phasic"] = kwargs_phasic
    report_info["kwargs_peaks"] = kwargs_peaks

    # Initialize refs list
    refs = []

    # 1. Cleaning
    # ------------
    report_info["text_cleaning"] = f"The raw signal, sampled at {sampling_rate} Hz,"
    if method_cleaning == "biosppy":
        report_info["text_cleaning"] += " was cleaned using the biosppy package."
    elif method_cleaning in ["default", "neurokit", "nk"]:
        report_info["text_cleaning"] += " was cleaned using the default method of the neurokit2 package."
    elif method_cleaning in ["none"]:
        report_info["text_cleaning"] += "was directly used without cleaning."
    else:
        report_info["text_cleaning"] += " was cleaned using the method described in " + method_cleaning + "."

    # 2. Phasic decomposition
    # -----------------------
    report_info["text_phasic"] = "The signal was decomposed into phasic and tonic components using"
    if method_phasic is None or method_phasic in ["none"]:
        report_info["text_phasic"] = "There was no phasic decomposition carried out."
    else:
        report_info["text_phasic"] += " the method described in " + method_phasic + "."

    # 3. Peak detection
    # -----------------
    report_info["text_peaks"] = "The cleaned signal was used to detect peaks using"
    if method_peaks in ["gamboa2008", "gamboa"]:
        report_info["text_peaks"] += " the method described in Gamboa et al. (2008)."
        refs.append("""Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci
        and electrophysiology. PhD ThesisUniversidade.""")
    elif method_peaks in ["kim", "kbk", "kim2004", "biosppy"]:
        report_info["text_peaks"] += " the method described in Kim et al. (2004)."
        refs.append("""Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term
      monitoring of physiological signals. Medical and biological engineering and computing, 42(3),
      419-427.""")
    elif method_peaks in ["nk", "nk2", "neurokit", "neurokit2"]:
        report_info["text_peaks"] += " the default method of the `neurokit2` package."
        refs.append("https://doi.org/10.21105/joss.01667")
    elif method_peaks in ["vanhalem2020", "vanhalem", "halem2020"]:
        report_info["text_peaks"] += " the method described in Vanhalem et al. (2020)."
        refs.append("""van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020).
      Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample
      Arousing Events Within an Experience Sampling Framework. European Journal of Personality.""")
    elif method_peaks in ["nabian2018", "nabian"]:
        report_info["text_peaks"] += " the method described in Nabian et al. (2018)."
        refs.append("""Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S. (2018). An
      Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data. IEEE
      journal of translational engineering in health and medicine, 6, 2800711.""")
    else:
        report_info[
            "text_peaks"
        ] = f"The peak detection was carried out using the method {method_peaks}."

    # References
    report_info["references"] = list(np.unique(refs))
    return report_info

def eda_process_nuevo(
    eda_signal, sampling_rate=1000, method="neurokit", report=None, **kwargs
):
    """
    Modificación para que el método de EDA_Tonic sea cvxOPT
    Cambio eda_methods() por eda_methods_nuevo()

    """
    # Sanitize input
    eda_signal = nk.signal_sanitize(eda_signal)
    methods = eda_methods_nuevo(sampling_rate=sampling_rate, method=method, **kwargs)

    # Preprocess
    # Clean signal
    eda_cleaned = nk.eda_clean(
        eda_signal,
        sampling_rate=sampling_rate,
        method=methods["method_cleaning"],
        **methods["kwargs_cleaning"],
    )
    if methods["method_phasic"] is None or methods["method_phasic"].lower() == "none":
        eda_decomposed = pd.DataFrame({"EDA_Phasic": eda_cleaned})
    else:
        eda_decomposed = nk.eda_phasic(
            eda_cleaned,
            sampling_rate=sampling_rate,
            method=methods["method_phasic"],
            **methods["kwargs_phasic"],
        )

    # Find peaks
    peak_signal, info = nk.eda_peaks(
        eda_decomposed["EDA_Phasic"].values,
        sampling_rate=sampling_rate,
        method=methods["method_peaks"],
        amplitude_min=0.1,
        **methods["kwargs_peaks"],
    )
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    # Store
    signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})

    signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

    if report is not None:
        # Generate report containing description and figures of processing
        if ".html" in str(report):
            fig = nk.eda_plot(signals, info, static=False)
        else:
            fig = None
        nk.create_report(file=report, signals=signals, info=methods, fig=fig)

    return signals, info


def bio_process_nuevo(
    ecg=None,
    rsp=None,
    eda=None,
    emg=None,
    ppg=None,
    eog=None,
    keep=None,
    sampling_rate=1000,
):
    """
    Modificación para que el método de EDA_Tonic sea cvxOPT
    Cambio eda_process() por eda_process_nuevo()

    """
    bio_info = {}
    bio_df = pd.DataFrame({})

    # Error check if first argument is a Dataframe.
    if ecg is not None:
        if isinstance(ecg, pd.DataFrame):
            data = ecg.copy()
            if "RSP" in data.keys():
                rsp = data["RSP"]
            else:
                rsp = None
            if "EDA" in data.keys():
                eda = data["EDA"]
            else:
                eda = None
            if "EMG" in data.keys():
                emg = data["EMG"]
            else:
                emg = None
            if "ECG" in data.keys():
                ecg = data["ECG"]
            elif "EKG" in data.keys():
                ecg = data["EKG"]
            else:
                ecg = None
            if "PPG" in data.keys():
                ppg = data["PPG"]
            else:
                ppg = None
            if "EOG" in data.keys():
                eog = data["EOG"]
            else:
                eog = None
            cols = ["ECG", "EKG", "RSP", "EDA", "EMG", "PPG", "EOG"]
            keep_keys = [key for key in data.keys() if key not in cols]
            if len(keep_keys) != 0:
                keep = data[keep_keys]
            else:
                keep = None

    # ECG
    if ecg is not None:
        print("Analizando ECG")
        ecg = nk.as_vector(ecg)
        ecg_signals, ecg_info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
        bio_info.update(ecg_info)
        bio_df = pd.concat([bio_df, ecg_signals], axis=1)

    # RSP
    if rsp is not None:
        print("Analizando resp")
        rsp = nk.as_vector(rsp)
        rsp_signals, rsp_info = nk.rsp_process(rsp, sampling_rate=sampling_rate)
        bio_info.update(rsp_info)
        bio_df = pd.concat([bio_df, rsp_signals], axis=1)

    # EDA
    if eda is not None:
        print("Analizando EDA")
        eda = nk.as_vector(eda)
        eda_signals, eda_info = eda_process_nuevo(eda, sampling_rate=sampling_rate)
        bio_info.update(eda_info)
        bio_df = pd.concat([bio_df, eda_signals], axis=1)

    # EMG
    if emg is not None:
        print("Analizando EMG")
        emg = nk.as_vector(emg)
        emg_signals, emg_info = nk.emg_process(emg, sampling_rate=sampling_rate)
        bio_info.update(emg_info)
        bio_df = pd.concat([bio_df, emg_signals], axis=1)

    # PPG
    if ppg is not None:
        print("Analizando PPG")
        ppg = nk.as_vector(ppg)
        ppg_signals, ppg_info = nk.ppg_process(ppg, sampling_rate=sampling_rate)
        bio_info.update(ppg_info)
        bio_df = pd.concat([bio_df, ppg_signals], axis=1)

    # EOG
    if eog is not None:
        print("Analizando EOG")
        eog = nk.as_vector(eog)
        eog_signals, eog_info = nk.eog_process(eog, sampling_rate=sampling_rate)
        bio_info.update(eog_info)
        bio_df = pd.concat([bio_df, eog_signals], axis=1)

    # Additional channels to keep
    if keep is not None:
        if isinstance(keep, pd.DataFrame) or isinstance(keep, pd.Series):
            keep = keep.reset_index(drop=True)
        else:
            raise ValueError("The 'keep' argument must be a DataFrame or Series.")

        bio_df = pd.concat([bio_df, keep], axis=1)

    # RSA
    if ecg is not None and rsp is not None:
        print("Analizando HRV_RSA")
        rsa = nk.hrv_rsa(
            ecg_signals,
            rsp_signals,
            rpeaks=None,
            sampling_rate=sampling_rate,
            continuous=True,
        )
        bio_df = pd.concat([bio_df, rsa], axis=1)

    # Add sampling rate in dict info
    bio_info["sampling_rate"] = sampling_rate

    return bio_df, bio_info

#%% 
subjects = ["01"]

for subject in subjects:
    fname = f'sub-{subject}/ses-A/eeg/sub-{subject}_ses-a_task-experiment_vr_non_immersive_eeg.fif'

    print(f'Leyendo archivo raw de {subject}')
    raw = mne.io.read_raw(fname, verbose=True, preload = True)
    
    print('Cargando datos raw')
    data = raw.load_data()    
    markers = raw.annotations
    
    conteo_anotaciones = markers.count() # --> acá cuento directo del archivo los markers, así se de primera mano cuantos hay
    
    df_data = data.to_data_frame(time_format=None)
    df_markers = markers.to_data_frame(time_format="ms")
    df_markers["onset"] = df_markers["onset"]/1000 # Para que quede en segundos
    
    print(df_data.columns)
    print(df_markers.columns)

#%%
    print('Extrayendo features con neurokit')
    df_bio, info_bio = bio_process_nuevo(ecg=df_data['ECG'], rsp=df_data['RESP'], eda=df_data['GSR'], keep=df_data['time'], sampling_rate=512)
        
    #%%
    # Guardo los df de interés para despues
    print(f'Guardando archivos en "sub-{subject}/ses-A"')
    df_bio.to_csv(f'sub-{subject}/ses-A/fisio_sujeto_{subject}.csv')
    df_markers.to_csv(f'sub-{subject}/ses-A/marcadores_sujeto_{subject}.csv')