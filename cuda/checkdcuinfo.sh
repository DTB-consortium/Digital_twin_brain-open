awk '{print $1}' hostfile | clush -w $0 - f 501 -b "hy_smi.py" > dcuinfo.log
