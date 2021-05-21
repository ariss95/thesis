import csv
import math
import numpy as np
#http://kdd.org/cupfiles/KDDCupData/1999/kddcup.data.zip
#link for kddcup

x = np.sqrt(6.0 / (64 + 256))
print(x)

path = "kddcup99/kddcup.data_10_percent_corrected"
file = open(path, "r")
csv_data = csv.reader(file, delimiter=",")
data = []
#row1 = [0,"tcp","http","SF",334,1684,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,9,0.00,0.00,0.00,0.00,1.00,0.00,0.33,0,0,0.00,0.00,0.00,0.00,0.00,0.00]

'''
the following list are all values from the non numeric collumns taken with this code:
for d in csv_data:
    if d[1] not in col1:
            col1.append(d[1])
        if d[2] not in col2:
            col2.append(d[2])
        if d[3] not in col3:
            col3.append(d[3])
'''
col1 = ['tcp', 'udp', 'icmp']
col2 = ['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp', 'eco_i', 'ntp_u', 'ecr_i', 'other', 'private', 'pop_3', 'ftp_data', 'rje', 'time', 'mtp', 'link', 'remote_job', 'gopher', 'ssh', 'name', 'whois', 'domain', 'login', 'imap4', 'daytime', 'ctf', 'nntp', 'shell', 'IRC', 'nnsp', 'http_443', 'exec', 'printer', 'efs', 'courier', 'uucp', 'klogin', 'kshell', 'echo', 'discard', 'systat', 'supdup', 'iso_tsap', 'hostnames', 'csnet_ns', 'pop_2', 'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn', 'netbios_dgm', 'sql_net', 'vmnet', 'bgp', 'Z39_50', 'ldap', 'netstat', 'urh_i', 'X11', 'urp_i', 'pm_dump', 'tftp_u', 'tim_i', 'red_i']
col3 = ['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'OTH', 'SH']
labels = []
for d in csv_data:
    d[1] = col1.index(d[1])
    d[2] = col2.index(d[2])
    d[3] = col3.index(d[3])
    if d[-1] == "normal.":
        d[-1] = 1
    else:
        d[-1] = 0
    labels.append(d.pop(len(d)-1))    
    data.append(d)
    #print(math.sqrt(len(d)))
print(len(data))
'''

'''