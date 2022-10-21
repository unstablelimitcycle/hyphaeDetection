from PIL import Image 
im = Image.open('/Users/lisarogers/Dropbox/Pictures for Lisa/Binary Pictures for Pixel Counting/62, 0mM/binarycolony0.jpg')
binaryimg = im.convert("1")
binaryimg.show()

unknownpixel = 0
black = 0
white = 0
red = 0
total = 0

for pixel in binaryimg.getdata():
    total+=1
    if pixel == 0:
        black += 1
    elif pixel ==255: 
        white +=1
    else:
        unknownpixel+=1
        print(str(pixel))
print('Total pixels = '+str(total),'White pixels = '+str(white), 'Black pixels = '+str(black), 'Red pixels = '+str(red), 'Unknown pixels = '+str(unknownpixel))
    #Change this to output in a spreadsheet, not just print