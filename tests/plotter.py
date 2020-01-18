from matplotlib import cm
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
from datetime import datetime

def plot_settings():
    fig = plt.figure(figsize=(3, 2), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    params = {'legend.fontsize': 3,
            'legend.handlelength': 3}
    plt.rcParams.update(params)
    plt.axis('off')
    return ax

def render_gif(timedir):

    images = []
    filenames = os.listdir(os.path.join('figures', timedir))
    imageio.plugins.freeimage.download()

    for filename in filenames:
        images.append(imageio.imread(os.path.join('figures', timedir, filename)))

    imageio.mimsave(os.path.join('figures', timedir, 'movie.gif'), images, format='GIF-FI', duration=0.25)

class OptimizationPlotter():

    def __init__(self, ax, sym_limits=1.5, plot_type='surface'):
        self.loss_func = None
        self.ax = plot_settings()
        self.plot_type = plot_type
        self.sym_limits=sym_limits
        # self.x_mesh, self.y_mesh, self.z_mesh = self.build_mesh()
        self.last_x, self.last_y, self.last_z = [],[],[]
        self.plot_time = datetime.now()
        self.active_surface = None

    def build_mesh(self):
        x_val = y_val = np.arange(-self.sym_limits, self.sym_limits, 0.005, dtype=np.float32)
        x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
        x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
        y_val_mesh_flat = y_val_mesh.reshape([-1, 1])
        z_val_mesh_flat = self.loss_func(torch.tensor(x_val_mesh_flat),
                                        torch.tensor(y_val_mesh_flat))                                
        z_val_mesh_flat = z_val_mesh_flat.numpy()
        return x_val_mesh, y_val_mesh, z_val_mesh_flat.reshape(x_val_mesh.shape)
    
    @staticmethod
    def clean_surface(ax):
        ax.clear()
        # fig = plt.figure(figsize=(3, 2), dpi=300)
        # ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        params = {'legend.fontsize': 3,
            'legend.handlelength': 3}
        plt.rcParams.update(params)
        plt.axis('off')

    def plot_surface(self):
        
        # remove the old surfaces if they exist
        if self.active_surface:
            self.active_surface.remove()

        levels = np.arange(-10, 1, 0.05)
        if self.plot_type == 'contour':
            self.active_surface = self.ax.contour(self.x_mesh, self.y_mesh, self.z_mesh, levels, alpha=.7, linewidths=0.4)
        elif self.plot_type == 'wireframe':
            self.active_surface = self.ax.plot_wireframe(self.x_mesh, self.y_mesh, self.z_mesh, alpha=.5, linewidths=0.4, antialiased=True)          
        elif self.plot_type == 'surface':
            self.active_surface = self.ax.plot_surface(self.x_mesh, self.y_mesh, self.z_mesh, alpha=.4, cmap=cm.coolwarm)
        # plt.draw()
    
    def loss_update(self, loss):
        self.loss_func = loss
        self.x_mesh, self.y_mesh, self.z_mesh = self.build_mesh()
        self.plot_surface()

    def adjust_camera_angle(self):
        lim3d_coeff = 0.75
        # 3d plot camera zoom, angle
        xlm = self.ax.get_xlim3d()
        ylm = self.ax.get_ylim3d()
        zlm = self.ax.get_zlim3d()
        self.ax.set_xlim3d(xlm[0] * lim3d_coeff, xlm[1] * lim3d_coeff)
        self.ax.set_ylim3d(ylm[0] * lim3d_coeff, ylm[1] * lim3d_coeff)
        self.ax.set_zlim3d(zlm[0] * lim3d_coeff, zlm[1] * lim3d_coeff)
        azm = self.ax.azim
        ele = self.ax.elev + 40
        self.ax.view_init(elev=ele, azim=azm)
    
    def update_plot(self, x_val, y_val, z_val):

        self.ax.scatter(x_val, y_val, z_val, s=3, depthshade=True, color='r')
        
        # draw a line from the previous value
        if not self.last_z:
            self.last_z.append(z_val)
            self.last_x.append(x_val)
            self.last_y.append(y_val)

        self.ax.plot([self.last_x[-1], x_val], 
                     [self.last_y[-1], y_val], 
                     [self.last_z[-1], z_val], 
                     linewidth=0.5, 
                     color='r')
        
        self.last_x.append(x_val.copy())
        self.last_y.append(y_val.copy())
        self.last_z.append(z_val.copy())

        if not os.path.exists(os.path.join('figures', str(self.plot_time))):
            os.makedirs(os.path.join('figures', str(self.plot_time)))

        plt.savefig(os.path.join('figures', str(self.plot_time), str(len(self.last_z)-2) + '.png'))
        plt.pause(0.0001)
    
    def plot_gif(self):
        render_gif(str(self.plot_time))
