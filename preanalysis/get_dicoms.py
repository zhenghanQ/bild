import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sub', help='particpant that you want to get dicoms for')
parser.add_argument('-u', '--user_acc', help='your user name')
parser.add_argument('-dd', '--dicom_dir', help='directory where dicoms are stored')
args = parser.parse_args()
sub = args.sub
user_acc = args.user_acc
dicom_dir = args.dicom_dir


subprocess.call(['ssh', '%s@sigma.mit.edu' % user_acc, '-t', 'sessionfinder',
		 '--s', '%s' % sub, '>', '%s.txt' % sub, ';', 'sed',
	 	 '-i', """'s/ /\\n/g'""", '%s.txt' % sub, ';', 'tail',
		 '-2', '%s.txt' % sub,'|', 'head', '-1', '>', 'dir.txt', ';'
		 'rm', '%s.txt' % sub])

if not os.path.isdir('%s//%s' % (dicom_dir, sub)):
    subprocess.call(['mkdir', '%s//%s' % (dicom_dir, sub)])

subprocess.call(['ssh', '%s@sigma.mit.edu' % user_acc, '-t', 'scp', 'dir.txt',
                 '%s@openmind7.mit.edu:%s//%s' % (user_acc, dicom_dir, sub), ';',
		 'rm', 'dir.txt'])

if not os.path.isdir('%s//%s/dicom' % (dicom_dir, sub)):
    os.makedirs('%s//%s/dicom' % (dicom_dir, sub))

with open('%s//%s/dir.txt' % (dicom_dir, sub)) as f:
    dir_ = f.readlines()
dir_ = str(dir_[0][:-1])

subprocess.call(['scp', '-r','%s@sigma.mit.edu:' % user_acc + dir_ , 
                 '%s//%s/dicom/' % (dicom_dir, sub)])

