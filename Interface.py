from tkinter import *
import os
from PIL import Image
root = Tk()
root.configure(background='blue')
IM_DIR = "gen_images/"
PADDING = 2

#img1 = PhotoImage(file="gen_images/seed6314603_0.png")
#img1Btn = Button(root, image=img1)
#img1Btn.image = img1
#img1Btn.grid(row=1, column=1)

def repopulate(f):
    ## PUT CODE HERE
    print(f)
    pass

def get_images(directory=IM_DIR):
    ims = []
    for f in os.listdir(directory):
        fl = os.path.join(directory, f)
        #img = Image.open(fl)
        #img = img.resize((100, 100), Image.ANTIALIAS)
        ims.append((PhotoImage(file=fl).subsample(10), fl))

    return ims

def on_click(f):
    def innerfunc():
        repopulate(f)
        image_grid()
        return f
    return innerfunc

def image_grid(directory=IM_DIR, COL_MAX=5):
    r = 0
    c = 0
    images_gui = []

    for im, f in get_images(directory)[:9]:
        if c == COL_MAX:
            c = 0
            r += 1
        button = Button(root, text='this one', image=im, command=on_click(f))
        button.image = im
        button.config(height = 100, width = 100)
        button.grid(row = r, column = c, sticky = W, pady = PADDING)
        images_gui.append(button)

        c += 1

    return images_gui

image_grid()
mainloop()
