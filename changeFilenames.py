import os

def changeFilenames(folder):
    path = os.path.dirname(folder)
    newpath = os.path.basename(path)

    for filename in os.listdir(folder):
        dst = os.path.join(path, newpath + '_' + filename)
        src = os.path.join(path,filename)
        os.rename(src, dst)