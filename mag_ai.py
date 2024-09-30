import warnings
from obspy import read, UTCDateTime
import glob
import numpy as np
import pandas as pd
from obspy.geodetics import locations2degrees
import matplotlib.pyplot as plt
import os
warnings.filterwarnings("ignore")

def cal_par(st_a):
    st_a.filter("highpass",freq=0.75)
    st_v = st_a.copy().integrate()
    st_v.filter("highpass",freq=0.75)
    st_d = st_v.copy().integrate()
    st_d.filter("highpass",freq=0.75)
    dt = st_a[0].stats.sampling_rate
    peakd = np.max(abs(st_d[0].data))
    pv = np.max(abs(st_v[0].data))
    pa = np.max(abs(st_a[0].data))
    r = np.trapz((st_v[0].data)**2, dx=dt)/np.trapz((st_d[0].data)**2, dx=dt)
    tauc = 2*np.pi/np.sqrt(r)
    tp = tauc * peakd
    tva = 2*np.pi*(pv/pa)
    piv = np.max(np.log10(abs(st_a[0].data*st_v[0].data)))
    iv2 = np.trapz((st_v[0].data)**2, dx=dt)
    cav = np.trapz(abs(st_a[0].data), dx=dt)
    cvad = np.sum(abs(st_d[0].data))
    cvav = np.sum(abs(st_v[0].data))
    cvaa = np.sum(abs(st_a[0].data))
    return pa, pv, peakd, r, tauc, tp, tva, piv, iv2, cav, cvad, cvav, cvaa



df = pd.read_csv("allevent_jawabarat.sh", delim_whitespace=True, names=['python','script','OT','lat','lon','dep','mag']) #generated from create_sh.py
df_sta = pd.read_excel("station.xlsx")
file_pga = open("mag_ml_2020_detik10.txt","w") #nama file sesuaikan
for i, eq in df.iterrows():
    dir_ms = str(eq['OT'])
    print(dir_ms)
    data_mseed = "/mnt/c/Users/naufa/OneDrive/Documents/AI Magnitude/201906/"+dir_ms+"/mseed" #data mseed
    
    if not os.path.exists(data_mseed):
        continue
    data_directory = "./gempajawabarat/"+dir_ms #directory data picking from pickp_uin.py
    
    if not os.path.exists(data_directory):
        continue
    eq_lat = float(eq['lat'])
    eq_lon = float(eq['lon'])
    depth = float(eq['dep'])
    mag = float(eq['mag'])
    ot_eq = UTCDateTime(eq["OT"])
    pick_file = glob.glob(f"{data_directory}/p_pick_times.txt")
    if len(pick_file)==0:
        continue
    try:
        pick_sta = pd.read_csv(f"{data_directory}/p_pick_times.txt", delim_whitespace=True, header=None, names=["kode","pickp"])
    except pd.errors.EmptyDataError:
        print("p_pick_times file is empty.")
        continue
    for j, sta in pick_sta.iterrows():
        code = sta["kode"]
        pickp = sta["pickp"]
        if pickp == '0' or pickp == 0:
            continue
        pickp = UTCDateTime("20"+pickp)
        mseed_file = glob.glob(f"{data_mseed}/{code}.mseed")
        if len(mseed_file) == 0:
            continue
        st = read(mseed_file[0])
        st.detrend("demean").merge(fill_value=0)
        st.detrend()
        st = st.select(channel="??Z").copy()
        metadata = df_sta[df_sta['Kode']==code]
        station_latitude = metadata['Lat'].values[0]
        station_longitude = metadata['Long'].values[0]
        dist = locations2degrees(eq_lat, eq_lon, station_latitude, station_longitude)
        dist *= 111
        st.trim(pickp-0.5,pickp+10)
        if len(st) == 0:
            continue
        pa, pv, peakd, r, tauc, tp, tva, piv, iv2, cav, cvad, cvav, cvaa = cal_par(st.copy())
        file_pga.write(f"{dir_ms} {code} {dist} {mag} {pa} {pv} {peakd} {r} {tauc} {tp} {tva} {iv2} {piv} {cav} {cvad} {cvav} {cvaa}\n")
    
file_pga.close()