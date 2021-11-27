import sys
import os
files = os.listdir(sys.argv[1])
for f in files:
    folder = os.path.join(sys.argv[1], f[:-4])
    os.makedirs(folder)
    os.rename(os.path.join(sys.argv[1], f), os.path.join(folder, f))
