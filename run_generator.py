#Generator for NVIDIA StyleGAN2 and Interactive interface

#-------------------------------------------------------------------------------
#Section: StyleGAN2 model imports
#-------------------------------------------------------------------------------
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import pretrained_networks
import matplotlib.pyplot as plt


#-------------------------------------------------------------------------------
#Section: Tkinter(interface) imports
#-------------------------------------------------------------------------------
from tkinter import *
import os
from PIL import Image, ImageTk
root = None
IM_DIR = "gen_images/"
PADDING = 10
u_select = -2

mode = None
OG = None

logo = None
drop_val = None

coefficient = .5

w = None

#-------------------------------------------------------------------------------
#Section: Interface drawing functions
#-------------------------------------------------------------------------------
def save():

    OG.save(dnnlib.make_run_dir_path('SuspectWitnessReport.png'))
    print('Saved!')
    global root
    root.destroy()
    global u_select
    u_select = -2
    return

def regenerate():
    print('Regenerating')
    global coefficient
    coefficient = variation_coefficient()
    global root
    root.destroy()
    global u_select
    u_select = -1
    return

def drop_menu_change(value):
    def inner(v):
        global coefficient
        coefficient = variation_coefficient()
        print(value.get())
        global mode
        mode = str(value.get())
        global root
        root.destroy()
        global u_select
        u_select = -1
        return
    return inner

class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master=master
        pad=3
        self._geom='200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
        master.bind('<Escape>',self.toggle_geom)
    def toggle_geom(self,event):
        geom=self.master.winfo_geometry()
        print(geom,self._geom)
        self.master.geometry(self._geom)
        self._geom=geom

def variation_coefficient():
    global w
    return w.get()/10.

def repopulate(f):
    ## PUT CODE HERE
    global coefficient
    coefficient = variation_coefficient()
    print('u_select set: '+ str(f-1))
    global u_select
    u_select = int(f-1)
    pass
    return

def on_click(f):
    def innerfunc():
        repopulate(f)
        global root
        root.destroy()
        return f
    return innerfunc


def image_grid(images, COL_MAX=4):
    global w
    #images = get_images(directory)
    global logo
    logo = PhotoImage(file='logo.png')
    img = Label(root, image=logo)
    img.grid(row = 0, column = 0, sticky=W, pady = PADDING)
    im = images[0]
    f = 0
    prev_img = Button(root, text='this one', image= im, command=on_click(f))
    prev_img.image = im
    prev_img.config(height = 314, width = 314, highlightthickness=0, relief=FLAT)
    prev_img.grid(row = 1, column = 0, sticky=W, pady = PADDING)

    descr = Button(root, text="Save current image", command=save)
    descr.config(width=60, height=1, highlightthickness=0, relief=FLAT, bg='white')
    descr.grid(row = 2, column = 0, sticky=W, pady = PADDING)
    r = 0
    c = 1
    images_gui = []

    count = 1
    for im in images[1:7]:
        f = count
        if c == COL_MAX:
            c = 1
            r += 1
        button = Button(root, text='this one', image=im, command=on_click(f))
        button.image = im
        button.config(height = 314, width = 314, highlightthickness=0, relief=FLAT)
        button.grid(row = r, column = c, sticky = W, pady = PADDING)
        images_gui.append(button)
        c += 1
        count +=1

    redo = Button(root, text="Regenerate Images", command=regenerate)
    redo.config(width=30, height=1, highlightthickness=0, relief=FLAT, bg='white')
    redo.grid(row = r + 1, column = 1, sticky=W, pady = PADDING)

    optionList = ["Random", "Aging", "Glasses", "African-American", "Asian", "Indian", "White",
    "Masculinity"]
    dropVar=StringVar()
    dropVar.set("Random") # default choice
    dropMenu = OptionMenu(root, dropVar, *optionList, command=drop_menu_change(dropVar))
    dropMenu.grid(row = r +1, column = 3, sticky = W, pady = PADDING)
    w = Scale(root, from_=-10, to=10, orient=HORIZONTAL, length=200)
    w.set(5)
    w.grid(row = r + 1, column = 2, sticky = W, pady = PADDING)

    return images_gui


#-------------------------------------------------------------------------------
#Section: Face Generation functions
#-------------------------------------------------------------------------------
def iter_gen(seeds, num_child, Gs, Gs_kwargs, noise_vars):
    global mode
    global OG
    global u_select
    global logo
    global coefficient

    z = None
    new_z = None

    z = np.zeros((1,512))

    rnd = np.random.RandomState(0)
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]

    while(True):
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        OG = PIL.Image.fromarray(images[0], 'RGB')
        OG_Img = OG.resize((314,314))
        print("parent")
        plt.imshow(OG_Img)
        #plt.show()
        children = []
        children_z = None
        for i in range(num_child):
            mod_z = np.random.randn(1, *Gs.input_shape[1:]) * coefficient
            mod_2 = 0
            if(mode is not None):
                mod_z = mod_z *.1
                if mode == 'African-American':
                    mod_2 = np.load('black_vec.npy') *600 * coefficient
                elif mode == 'Indian':
                    mod_2 = np.load('indian_vec.npy') * 40 * coefficient
                elif mode == 'Glasses':
                    mod_2 = np.load('eyeglasses_vec.npy') * 20 * coefficient
                elif mode == 'Aging':
                    mod_2 = np.load('gray_vec.npy') * 20 * coefficient
                elif mode == 'Asian':
                    mod_2 = np.load('asian_vec.npy') * 20 * coefficient
                elif mode == 'White':
                    mod_2 = np.load('white_vec.npy') * 8 * coefficient
                elif mode == 'Masculinity':
                    mod_2 = np.load('m_vec.npy') * 2 * coefficient

                mode = None
            #mod2 = np.load('black_vec.npy') *100
            #print(mod_z)
            new_z = z + mod_z + mod_2# + mod_3
            image = Gs.run(new_z, None, **Gs_kwargs)
            child = PIL.Image.fromarray(image[0], 'RGB').resize((314,314))
            children.append(child)
            if children_z is None:
                children_z = new_z
            else:
                children_z = np.concatenate((children_z, new_z))

        count = 1
        plt.subplot(2,num_child//2+1,1)
        plt.imshow(OG_Img)
        #plt.set_title('OG')
        for i in children:
            plt.subplot(2,num_child//2+1,count+1)
            plt.imshow(i)
            #plt.set_title(str(count-1))
            count = count+1
        #plt.show()
        count = 1

        global root

        root = Tk()
        #root.configure(background='white')
        app=FullScreenApp(root)
        logo = PhotoImage(file='logo.png')
        root.config(padx=10, pady=10)

        c = []
        c.append(ImageTk.PhotoImage(OG_Img, master = root))
        for i in children:
            c.append(ImageTk.PhotoImage(i, master=root))

        print(len(c))
        image_grid(c)
        mainloop()
        print("Enter Index or exit:\n")
        usr = u_select
        if(usr == -2):
            return z
        elif(int(usr)==-1):
            z = z
        elif(int(usr)<0 or int(usr)>=num_child):
            print("Enter valid index\n")
            usr=input()
        else:
            z = children_z[int(usr)]
            z = z.reshape((1,512))
            #print(z)
        u_select = -2
        print(str(int(usr)))


def generate_images(network_pkl, seeds, truncation_psi):

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = True
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    iter_gen(seeds,6, Gs, Gs_kwargs,noise_vars)

    '''#Old generate_images
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    z = None
    count = 1002
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        #print(z)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('gen_img%04d.png' % count))
        np.save(dnnlib.make_run_dir_path('z_vector%04d.npy'%count),z)
        count+=1
        for i in range(2):
            mod_z = np.random.randn(1, *Gs.input_shape[1:]) * .3 #+(np.ones((1, *Gs.input_shape[1:]))*2000000)
            #print(mod_z)
            new_z = z + mod_z
            #print(z-new_z)
            image = Gs.run(new_z, None, **Gs_kwargs)
            PIL.Image.fromarray(image[0], 'RGB').save(dnnlib.make_run_dir_path('seed'+str(seed)+'_'+str(i)+ '.png'))'''



#-------------------------------------------------------------------------------
#Section: Helpers
#-------------------------------------------------------------------------------
def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]

#-------------------------------------------------------------------------------
#Section: Helpers
#-------------------------------------------------------------------------------

_examples = '''examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=6000-6025 --truncation-psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
'''


def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')

    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=False, default='stylegan2-ffhq-config-f.pkl')
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=False, default = '1-2')
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)',required = False, default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing video')
    parser_style_mixing_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-images': 'run_generator.generate_images',
        'style-mixing-example': 'run_generator.style_mixing_example'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)


if __name__ == "__main__":
    main()

#-------------------------------------------------------------------------------
#Section: Useless
#-------------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.png' % (row_seed, col_seed)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path('grid.png'))
