This is Python script to handle seismic data specifically with .mseed datasets.
This script handles auto P-wave picking, and handles directory management to distinguish which mseed has trigger event and has no trigger event. This is all stored in pickp_uin.py. 
Then the script will generate seismic parameters from the trigger event data. This is the function of mag_ai.py.
create_sh.py is the script to generate bash file (.sh) which contain the script to run and read every data in dataset folder that will be used in mag_ai.py
