# -*- coding: utf-8 -*- 
# @Time : 2021/5/30 13:42 
# @Author : lepold
# @File : ln.py.py


import os
import time

def progress_bar(progress, time):
    """ Print progress bar to console output in the format
    Progress: [######### ] 90.0% in 10.22 sec

    Parameters
    ----------
    progress : float
        Value between 0 and 1.
    time : float
        Elapsed time till current progress.
    """

    print("\r| Progress: [{0:10s}] {1:.1f}% in {2:.0f} sec".format(
        '#' * int(progress * 10), progress * 100, time), end='')
    if progress >= 1:
        print("\r| Progress: [{0:10s}] {1:.1f}% in {2:.2f} sec".format(
            '#' * int(progress * 10), progress * 100, time))

    return

subject = 'dti_distribution_86000m_d1000_with_map'
prefix = "/work/home/bujingde/project/zenglb/Digital_twin_brain/data/newnewdata_with_brainsteam"
new_subject = "dti_distribution_86000m_d1000_with_map_split"
os.makedirs(os.path.join(prefix, subject, 'ensembles'), exist_ok=True)
N = 14000
ensembles = 1
computation_start = time.time()
for i in range(N):
    progress = i / N
    progress_bar(progress, time.time() - computation_start)
    for j in range(ensembles):
        os.system("ln -s %s/%s/module/uint8/block_%d.npz %s/%s/ensembles/block_%d.npz"%(prefix, subject, i, prefix, new_subject, j*N+i))
