from tqdm import tqdm
import argparse
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('--indir', help = "The path of the flat folder")
parser.add_argument('--outdir', help = "The path of the output folder")
args = parser.parse_args()

if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
for idx, img_name in enumerate(os.listdir(args.indir)):
	print("%d steps reached"%idx)
	if idx == 50000:
		break
	try:
		img = cv2.imread(os.path.join(args.indir, img_name))
		cv2.imwrite(os.path.join(args.outdir, img_name[:-5] + '.jpg'), img) 
	except:
		continue
