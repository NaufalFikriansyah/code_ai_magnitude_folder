import os, sys, shutil, warnings
from obspy import read, UTCDateTime
from obspy.taup import TauPyModel
import glob, math
import seisbench.models as sbm
model_ps = sbm.PhaseNet.from_pretrained('stead')
model_eqt = sbm.EQTransformer.from_pretrained('original')
import numpy as np
import pandas as pd
from obspy.geodetics import locations2degrees
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
model = TauPyModel(model="iasp91")  

def get_P_pick(eq_lat, eq_lon, ot, station_latitude, station_longitude, depth=0):
    dist = locations2degrees(eq_lat, eq_lon, station_latitude, station_longitude) 
    dist = float(dist)
    depth = float(depth)
    arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist,
                                      receiver_depth_in_km=0, phase_list=['p',"P"])
    timep = arrivals[0].time
    return ot+timep


def find_similar_picks(eqt_p_picks, pn_p_picks, threshold=1.0):
    similar_picks = []
    for eqt_pick in eqt_p_picks:
        similar_picks.append(eqt_pick.timestamp)
    for pn_pick in pn_p_picks:
        similar_picks.append(pn_pick.timestamp)
    return similar_picks

def get_pick_ai(st):
    picks_ps = model_ps.classify(st)
    picks_eqt = model_eqt.classify(st)
    pick_ps = picks_ps.picks.select(phase='P')
    pick_eqt = picks_eqt.picks.select(phase='P')
    if len(pick_ps) == 0:
        list_ps = np.array([])
    else:
        list_ps = []
        for pick in pick_ps:
            list_ps.append(pick.start_time)
        list_ps = np.array(list_ps)
    if len(pick_eqt) ==0:
        list_eqt = np.array([])
    else:
        list_eqt = []
        for pick in pick_ps:
            list_eqt.append(pick.start_time)
        list_eqt = np.array(list_eqt)
    return list_ps, list_eqt

def plot_trig(st, timepick, outf):
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    stream = st.slice(timepick-10, timepick+1.5*60).copy()
    ax.plot(stream[-1].times(), stream[-1].data, 'k', label=stream[-1].stats.channel)
    second_pick = timepick - stream[-1].stats.starttime
    ax.axvline(x=second_pick,color='red',label='auto pick')
    ax.set_xlim([0, max(stream[-1].times())])
    # ax.set_xlim([second_pick-5, second_pick+90])
    ax.legend(loc=2)
    fig.savefig(outf)
    plt.close()
    
def plot_nottrig(stream, pn_preds, eqt_preds, outf):
    color_dict = {"P": "C0", "S": "C1", "Detection": "C2"}
    fig, ax = plt.subplots(5, 1, figsize=(15, 7), sharex=True, gridspec_kw={'hspace' : 0.05, 'height_ratios': [1, 1, 1, 1, 1]})
    
    for i, preds in enumerate([eqt_preds, pn_preds]):
        model1 = None
        for pred_trace in preds:
            model1, pred_class = pred_trace.stats.channel.split("_")
            if pred_class == "N":
                # Skip noise traces
                continue
            c = color_dict[pred_class]
            ax[i + 3].plot(pred_trace.times(), pred_trace.data, label=pred_class, c=c)
        if model1:
            ax[i + 3].set_ylabel(model1)
        ax[i + 3].legend(loc=2)
    for i in range(len(st)):
        ax[i].plot(stream[i].times(), stream[i].data / np.amax(stream[i].data), 'k', label=stream[i].stats.channel)
        ax[i].set_xlim(0, max(stream[-1].times()))
        ax[i].legend(loc=2)
    ax[1].set_ylabel('Normalised Amplitude')
    ax[-1].set_xlabel('Time [s]')
    fig.savefig(outf)
    plt.close()

read_event = pd.read_csv("/mnt/c/Users/naufa/OneDrive/Documents/AI Magnitude/201906/events_2018-2022.txt", delimiter='|', skipinitialspace=True ,names=["id", "ot", "Latitude", "Longitude", "Depth", "Mag", "Type M"])
#data event
for i,event in read_event.iterrows():
    event_time = UTCDateTime(event["ot"]) #UTCDateTime("20230520193313")
    eq_lat = event["Latitude"] #-7.41708231
    eq_long = event["Longitude"] # 107.2624969
    eq_depth = event["Depth"] #81
    event_id = event_time.strftime("%Y%m%d%H%M%S")
    print(event_id)
    data_directory = "/mnt/c/Users/naufa/OneDrive/Documents/AI Magnitude/201906/"+event_id+'/mseed' #data meseed
    
    if not os.path.exists(data_directory):
        continue
    out_dir = "./gempajawabarat/2019_3"+event_id #directory output

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    output_file = f"{out_dir}/p_pick_times.txt"

    files = glob.glob(f"{data_directory}/*.mseed")
    if len(files)==0:
        sys.exit("No MSEED File")
    trig_dir = f"{out_dir}/plot_trigger"
    nottrig_dir = f"{out_dir}/plot_nottrigger"

    if not os.path.exists(trig_dir):
        os.makedirs(trig_dir)
    else:
        shutil.rmtree(trig_dir)
        os.makedirs(trig_dir)
        
    if not os.path.exists(nottrig_dir):
        os.makedirs(nottrig_dir)
    else:
        shutil.rmtree(nottrig_dir)
        os.makedirs(nottrig_dir)
        


    df = pd.read_excel("station.xlsx")

    outf = open(output_file, 'w')
    for file in files:
        st = read(file)
        st.merge(fill_value=0)
        code = st[0].stats.station
        sta = df[df['Kode']==code]
       
        station_latitude = sta['Lat'].values[0]
        station_longitude = sta['Long'].values[0]
        list_ps, list_eqt = get_pick_ai(st)
        fix_pick = find_similar_picks(list_eqt, list_ps, threshold=2.0)
        pick_sta = None
        pick_theo = None
        if fix_pick:
            pick_theo = get_P_pick(float(eq_lat), float(eq_long), event_time, float(station_latitude), float(station_longitude), depth=eq_depth)
            for i, pick in enumerate(fix_pick):
                pick = UTCDateTime(pick)
                if abs(pick-pick_theo) <= 11:
                    if pick_sta:
                        if abs(pick-pick_theo) < abs(pick_sta-pick_theo):
                            pick_sta = pick
                    else:
                        pick_sta = pick
        if pick_sta:
            outf.write(f"{code} {pick_sta.strftime('%y-%m-%dT%H:%M:%S')}\n")
            plot_trig(st, pick_sta, f"{trig_dir}/{code}.png")
        else:
            pn_preds = model_ps.annotate(st)
            eqt_preds = model_eqt.annotate(st)
            if fix_pick:
                print(pick_theo, fix_pick, code)
                try:
                    plot_nottrig(st, pn_preds, eqt_preds, f"{nottrig_dir}/{code}_theo.png")
                except ValueError as e:
                    print(e)
            else:
                try:
                    plot_nottrig(st, pn_preds, eqt_preds, f"{nottrig_dir}/{code}.png")
                except ValueError as e:
                    print(e)
            outf.write(f"{code} 0\n")
    outf.close()
    print(f"Hasil telah disimpan di: {output_file}")
