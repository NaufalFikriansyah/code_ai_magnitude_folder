import pandas as pd
import os
from obspy import UTCDateTime

read_event = pd.read_csv("/mnt/c/Users/naufa/OneDrive/Documents/AI Magnitude/201906/events_2018-2022.txt", delimiter='|', skipinitialspace=True ,names=["id", "ot", "Latitude", "Longitude", "Depth", "Mag", "Type M"])
output_dir = "." 
os.makedirs(output_dir, exist_ok=True)

sh_content = ""
for i, event in read_event.iterrows():
  event_time = UTCDateTime(event["ot"])
  eq_lat = event["Latitude"]
  eq_long = event["Longitude"]
  eq_depth = event["Depth"]
  eq_mag = event["Mag"]
  event_id = event_time.strftime('%Y%m%d%H%M%S')
  print(type(event_id))
  sh_content += f"python pick_uin.py {event_id} {eq_lat} {eq_long} {eq_depth} {eq_mag}\n"

with open(os.path.join(output_dir, "allevent_jawabarat.sh"), 'w') as sh_file:
  sh_file.write(sh_content)