#from libforGUI import PySimpleGUI as sg 
import PySimpleGUI as sg
import tkinter as tk 
from os import listdir
from hyphaeDelityFunctions import hyphaedelity

image_icon = "sprites/icon_image.png"
#Aesthetic Window Things
sg.theme('DarkTeal7')

def main():       
    # Define Window Contents
    layout = [  
            [sg.Text("Select a folder of image files:  ", size=(22,1), font='Helvetica', justification='left'), 
            sg.In(size=(15,1), enable_events=True, key='-FOLDER1-'), 
            sg.FolderBrowse()],

            # [sg.Text("Select an image filetype: ", size=(20,1), font='Helvetica', justification='left'), 
            # sg.Combo(['.jpg','-'], default_value='jpg', enable_events =True, key="-COMBO1-")],
            # #sg.Combo(['jpg', 'tiff', 'png','eps','bmp','RAW'], default_value='jpg', enable_events =True, key="-COMBO1-")],

            # [sg.Text("Select desired output:", size=(20,1), font='Helvetica', justification='left'), 
            # sg.Combo(['All Pictures and Statistics','-'], default_value='All Pictures and Statistics', key='-COMBO2-', enable_events =True)],
            # #sg.Combo(['All Pictures and Statistics','Processed Image','Contours','CSV with Statistics'], default_value='All Pictures and Statistics', key='-COMBO2-', enable_events =True)],

            [sg.Text("Select folder to save files to:", size=(22,1), font='Helvetica', justification='left'), 
            sg.In(size=(15,1), enable_events=True, key='-FOLDER2-'), 
            sg.FolderBrowse()],
            
            [sg.Button('Run Image Analysis'),sg.Button('Quit')],    
            ]   

    # Create Window
    window = sg.Window('hyphaeDelity Yeast Image Analysis',layout, size=(400,150), icon=image_icon)

    #Run Main Loop
    while True:
        event,values = window.read()

        #Closing Window
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break
        
        #Choosing Folder: Choose a folder to read/import the files from
        if event =='-FOLDER1-':
            folder_path = values['-FOLDER1-']
        
        # Choosing Folder to save files to
        if event == '-FOLDER2-':
            file_path = values['-FOLDER2-']

        #Starting Image Analysis     
        if event == 'Run Image Analysis':
            hyphaedelity(folder_path,'.jpg',file_path) 
            #Put in text that indicates analysis complete
            break
        #Close Window once program is complete
    window.close()

if __name__ == '__main__':
    main()
#For Extra Inputs and Fancier GUI
    # #Choosing Filetype: When a filetype is chosen, only files of that type are read in
    # if event == '-COMBO1-':
    #     filenames = listdir(folder1)
    #     try:  
    #         # get list of files from folder
    #         file_list = listdir(folder)
    #     except:
    #         folder = []
    #     #User chooses filetype with selection of one from Combo list
    #     #if combobox choice is jpg, set filetype to jpg
    #     if values[-COMBO-]== 'jpg':
    #         filetype =    
    #     #get filetype chosen   
    #     if filetype.endswith('.jpg')==True: 
    #         stringsforCSV = [f for f in filenames if f.endswith('.jpg')]
    #     elif filetype.endwith('.tiff')==True:
    #         stringsforCSV = [f for f in filenames if f.endswith('.tiff')]

    # #Choosing Outputs: Output just stats, just pix, both stats and pix
    # if event =='-COMBO2-':
    #    if values[-COMBO2-]=='jpg'
 