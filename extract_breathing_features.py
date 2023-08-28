from breathing_features import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Extract the breathong features of the .mp4 file')
# Add the input parameters to the parser
parser.add_argument('--filename', type=str, help='the name of the .mp4 file to parse')
parser.add_argument('--is_normal_breath', action='store_true', help='set this flag if the breath is normal and not deep')
args = parser.parse_args()

# Get the values of the input parameters
filename = args.filename
is_normal_breath = args.is_normal_breath


y, signal_1, signal_2, significant_area, rr, rr_std = get_valid_parts(filename)
exhale, inhale = annotate_parts(y, signal_1, signal_2, significant_area)

if is_normal_breath:
    f = normal_breathing_features(inhale, exhale)
    
    print ("Breathing Features")
    print (f"RR: {round (rr)} bps")
    if len(f['inhale/exhale']) != 0:
        print (f"inhale/exhale:", round(np.mean(f['inhale/exhale']), 4), "±", round(np.std(f['inhale/exhale']), 4))
    else:
        print (f"inhale/exhale: -")
    
    if len(f['inhale/total']) != 0:
        print (f"inhale/total:", round(np.mean(f['inhale/total']), 4), "±", round(np.std(f['inhale/total']), 4))
    else:
        print (f"inhale/total: -")

plt.plot(y)
plt.plot(inhale, label="inhale")
plt.plot(exhale, label="exhale")
plt.plot(significant_area, label="significant area for the calculation")
plt.legend(loc="lower right")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

