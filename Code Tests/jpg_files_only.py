import glob
import shutil 
import os

src_dir = '/Users/lisarogers/PythonStuff/allImages/23022021/0uM/'
dst_dir = '/Users/lisarogers/PythonStuff/allImages/jpgonly23022021/'
for jpgfile in glob.iglob(os.path.join(src_dir, '*.jpg')):
    shutil.copy(jpgfile,dst_dir)