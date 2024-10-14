# -*- coding: utf-8 -*-
import os
import sys
import ast
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
import scipy.signal as sc

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

#%%
subjects = ["01"]#,"02","03","04","05","06"]

seleccion = ["EDA_Clean","EDA_Phasic","EDA_Tonic","SCR_Peaks","SCR_Amplitude",
             "ECG_Clean","ECG_Rate", "ECG_R_Peaks",
             "RSP_Clean","RSP_Amplitude","RSP_Rate","RSP_RVT",
             "time"]

for subject in subjects:
    df_fisio = pd.read_csv(f'sub-{subject}/ses-A/fisio_sujeto_{subject}.csv',usecols=seleccion)
    df_markers = pd.read_csv(f'sub-{subject}/ses-A/marcadores_sujeto_{subject}.csv').drop("Unnamed: 0", axis=1)
    
    # Acá intenta mergear los datos y los markers según la columna time y onset (se te van a sumar filas)
    df_markers.rename(columns={"onset": "time"}, inplace=True)
    merged_df = pd.merge(df_fisio, df_markers, on='time', how='outer')
    
    #%%
    # Agarrar los tiempos de los videos
    video_start_times = merged_df.loc[merged_df["description"]=="video_start","time"]
    video_end_times = merged_df.loc[merged_df["description"]=="video_end","time"]
    
    video_dict = {}
    
    # Para cada indice y time en video_start_times
    for i, t in enumerate(video_start_times):
        print(i)
        
        # El slice se guarda con una llave en el diccionario
        try:
            video_dict[i+1] = merged_df[(merged_df["time"] > t) & (merged_df["time"] < video_end_times.iloc[i])]
        
        # Por si hay más inicios de video que finales (quizas pasa con el último)
        except IndexError:
            video_dict[i+1] = merged_df[merged_df["time"] > t]
    
    # Largo aproximado de cada video
    for n in range(len(video_dict)):
        print(f'Largo video {n+1}: {len(video_dict[n+1])}')
        # Estaría bueno compararlo a simple vista con el orden de la presentación
        # y ver si coinciden las duraciones a grosso modo
    
    #%% Leo el df_beh y extraigo las columnas que quiero
    
    df_beh = pd.read_csv(f'sub-{subject}/ses-A/beh/sub-{subject}_ses-A_task-Experiment_VR_non_immersive_beh.csv')
    
    id_videos = df_beh.loc[4:,"id"]
    valence_videos = df_beh.loc[4:,"stimulus_type"]
    annotations = [ast.literal_eval(continuous_annotation) for continuous_annotation in df_beh.loc[4:,"continuous_annotation"]]
    dimension_annotated = df_beh.loc[4:,"dimension"]
    
    for i, video in enumerate(id_videos):
        df = video_dict[i+1]
        df["video_id"] = video
        df["stimulus_type"] = list(valence_videos)[i]
        df["dimension_annotated"] = list(dimension_annotated)[i]
        df["annotation"] = annotations[i]
        df["subject_id"] = subject
    
    df_final = pd.concat([df for df in video_dict.values()],ignore_index=True)
    df_final.to_csv(f'sub-{subject}/ses-A/df_sub-{subject}_final.csv')

#%%




# Acá abajo hay todas pruebas de cosas
#%% Para comparar ambas formas de interpolar (lineal vs previous)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Sample data: replace this with your actual data
original_samples = np.linspace(0, 1, 2332)  
variable_values = np.array(b[0])  # Example variable ranging from -1 to 1

# New sample points
new_sample_points = np.linspace(0, 1, 16904)

# Create interpolation function
interpolation_function_previous = interp1d(original_samples, variable_values, kind='previous')

interpolation_function_linear = interp1d(original_samples, variable_values, kind='linear')

# Interpolate


# Plotting for visualization (optional)
plt.figure(figsize=(12, 6))
plt.plot(original_samples, variable_values, 'o', label='Original samples')
plt.plot(new_sample_points, interpolation_function_previous(new_sample_points), '-', label='Interpolated previous')
plt.plot(new_sample_points, interpolation_function_linear(new_sample_points), '-', label='Interpolated linear')
plt.title('Interpolation of Variable from 2332 to 16904 Samples')
plt.xlabel('Sample Points')
plt.ylabel('Variable Values')
plt.legend()
plt.grid()
plt.show()

#%% Para comparar ambas formas de interpolar pero ver punto por punto


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Sample data: replace this with your actual data
original_samples = np.linspace(0, 1, 2332)  
variable_values = np.array(b[0])  # Example variable ranging from -1 to 1

# New sample points
new_sample_points = np.linspace(0, 1, 16904)

# Create interpolation function
interpolation_function_previous = interp1d(original_samples, variable_values, kind='previous')

interpolation_function_linear = interp1d(original_samples, variable_values, kind='linear')

# Interpolate


# Plotting for visualization (optional)
plt.figure(figsize=(12, 6))
plt.plot(original_samples, variable_values, 'o', label='Original samples')
plt.plot(new_sample_points, interpolation_function_previous(new_sample_points), '*', label='Interpolated previous')
plt.plot(new_sample_points, interpolation_function_linear(new_sample_points), '*', label='Interpolated linear')
plt.title('Interpolation of Variable from 2332 to 16904 Samples')
plt.xlabel('Sample Points')
plt.ylabel('Variable Values')
plt.legend()
plt.grid()
plt.show()



#%% Esta parte es de otro archivo que copie y pegué acá porque pienso que puede servir
    events = nk.events_create(df_markers["onset"], event_durations=df_markers["duration"], event_conditions=df_markers["description"])
    nk.events_plot([x//2 for x in events["onset"]], signal=seleccion, color='red', linestyle='--') 
    

