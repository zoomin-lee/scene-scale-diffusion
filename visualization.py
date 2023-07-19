
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import argparse
import numpy as np
import yaml
import struct

parser = argparse.ArgumentParser()
parser.add_argument('--M', default='scene-scale-diffusion') # VQVAE, multinomial_diffusion
parser.add_argument('--Driver', default='D')
parser.add_argument('--frame', default='0')
parser.add_argument('--file', default='result_')
parser.add_argument('--folder', default='Completion')
parser.add_argument('--model', default='image_init8_concat_att')
parser.add_argument('--name', default='Semantic Scene Completion')
parser.add_argument('--invalid', default = False)

class SpheresApp:
    MENU_SCENE = 1
    MENU_BEFORE = 2
    MENU_QUIT = 3

    def __init__(self, opt):
        self._id = 0
        self.opt = opt
        
        self.window = gui.Application.instance.create_window("Semantic Scene Completion", 1500, 1000)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.set_sun_light(
            [-0.577, 0.577, -0.577],  # direction
            [1, 1, 1],  # color
            60000)  # intensity
        
        self.scene.scene.scene.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([64, 64, -60], [64, 64, 60])
        
        self.scene.setup_camera(60, bbox, [0, 0, 1])
        self.window.add_child(self.scene)


        if gui.Application.instance.menubar is None:
            
            debug_menu = gui.Menu()
            debug_menu.add_item("Next Scene", SpheresApp.MENU_SCENE)
            debug_menu.add_separator()
            debug_menu.add_item("Before Scene", SpheresApp.MENU_BEFORE)
            debug_menu.add_separator()
            debug_menu.add_item("Quit", SpheresApp.MENU_QUIT)
            menu = gui.Menu()
            menu.add_menu("SSC", debug_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the menu item is activated.
        self.window.set_on_menu_item_activated(SpheresApp.MENU_SCENE,self._on_menu_scene)
        self.window.set_on_menu_item_activated(SpheresApp.MENU_QUIT,self._on_menu_quit)
        self.window.set_on_menu_item_activated(SpheresApp.MENU_BEFORE,self._on_menu_before)

    def _on_menu_before(self):
        self._id -= 1
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"

        if self.opt.file == 'input_':
            points = get_input(self.opt)
        else :
            points, colors = get_voxel(self.opt)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if (self.opt.file != 'input_'):
            pcd.colors = o3d.utility.Vector3dVector(colors/255)
        self.scene.scene.clear_geometry()
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
        self.scene.scene.add_geometry("scene" + str(self._id), voxel_grid, mat)
        print(self.opt.frame)
        self.opt.frame = str(int(self.opt.frame)-1)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_scene(self):
        self._id += 1
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"

        if self.opt.file == 'input_':
            points = get_input(self.opt)
        else :
            points, colors = get_voxel(self.opt)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if (self.opt.file != 'input_'):
            pcd.colors = o3d.utility.Vector3dVector(colors/255)
        self.scene.scene.clear_geometry()
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
        self.scene.scene.add_geometry("scene" + str(self._id), voxel_grid, mat)
        print(self.opt.frame)
        self.opt.frame = str(int(self.opt.frame)+1)

def get_voxel(opt):

    if opt.invalid :
        invalid_path = opt.Driver+':/'+opt.M+ '/result/' + opt.model +'/Invalid/invalid_'+ opt.frame +'.txt'
        invalid_points = np.loadtxt(invalid_path, delimiter=' ')
        invalid_colors = np.full(len(invalid_points,), 0)
                
        point_cloud_path = opt.Driver+':/'+opt.M+ '/result/' + opt.model +'/' + opt.folder +'/'+ opt.file + opt.frame +'.txt'
        points_colors = np.loadtxt(point_cloud_path, delimiter=' ')
        points = points_colors[:, 1:]
        colors = points_colors[:, 0]
        
        points = np.concatenate((invalid_points, points), axis=0)
        colors = np.concatenate((invalid_colors, colors), axis=0)
        
        points, index = np.unique(points, return_index=True, axis=0)
        colors = colors[index, ...]
        
    else :
        point_cloud_path = 'C:/Users/jumin/Dataset/result_319_110.txt'
        points_colors = np.loadtxt(point_cloud_path, delimiter=' ')
        points = points_colors[:, 1:]
        colors = points_colors[:, 0]

    if opt.dataset == 'carla' : 
        config_file = 'C:/Users/jumin/Dataset/carla.yaml'
        config = yaml.safe_load(open(config_file, 'r'))
        color_map = config["remap_color_map"]
    
    color = np.asarray([color_map[c] for c in colors])

    return points, color

def get_input(opt):
    point_cloud_path=opt.Driver+':/'+opt.M+'/result/' + opt.model +'/Invalid/invalid_' + opt.frame +'.txt'
    points_colors = np.loadtxt(point_cloud_path, delimiter=' ')
    points = points_colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return points

def main(opt):
    gui.Application.instance.initialize()
    SpheresApp(opt)
    gui.Application.instance.run()

if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)