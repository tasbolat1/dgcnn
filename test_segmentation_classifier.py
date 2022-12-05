from signal import SIGUSR2
import torch
from model import DGCNN_partseg
from utils.points import regularize_pc_point_count
import numpy as np
import pickle
import torch.nn as nn
import trimesh
from utils.visualization import gripper_bd

torch.manual_seed(1)
torch.cuda.manual_seed(1)


clusters_info = { # handover, non-handover
    'spatula': [23,22]
}

def get_real_pc(B, cat, trial):
    with open(f'semantic/{cat}/trial{trial}/{cat}.pkl', 'rb') as fp:
        data = np.array(pickle.load(fp))
        pc = data[:,:3]
    pc = regularize_pc_point_count(pc, 2048, False)

    print(pc)

    pc -= np.mean(pc, axis=0)
    pc[:, 1] +=0.01
    
    quaternion = None
    translation = None

    euler = [120,0,0]
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler( "xyz",euler, degrees=True) 
    transform = np.eye(4)
    transform[:3,:3] = r.as_matrix()
    transform[:3,3] = translation


    pc = pc @ np.linalg.inv(transform[:3,:3])
        
    # final_pts += np.expand_dims(transform[]:3,3], axis=1)


    # print(pc.shape)
    pc = np.expand_dims(pc, axis=0)
    # print(pc.shape)
    pc = torch.from_numpy(pc.astype(np.float32))
    pc = pc.permute(0, 2, 1)
    return pc


device = torch.device("cuda:0")

model = DGCNN_partseg(seg_num_all=50).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('pretrained/model.partseg.t7'))
model = model.to(device)
model.eval()

def get_segmented_labels(pc, label_one_hot):
    # in: label_one_hot [B,16], pc [B,3,2048]
    # out: pred [B, 2048]
    with torch.no_grad():
        seg_pred = model(pc, label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
    return pred

# prepare inputs
B = 1 # batch_size
label_one_hot = torch.zeros((B, 16))
label_one_hot[:,7] = 1

# pc = torch.rand([B,3,2048])
pc = get_real_pc(B, 'black_spatula',3)

# # prepare auxiliary inputs
# seg = torch.zeros((B,2048)) + 7


pc, label_one_hot = pc.to(device), label_one_hot.to(device)
print(pc.shape, label_one_hot.shape)
pred = get_segmented_labels(pc, label_one_hot)

print(torch.unique(pred))
pred_np = pred.cpu().numpy()[0]

print(pred_np.shape)

colors = np.zeros([2048, 4])
colors[:,3] = 255
print(np.sum(pred_np==22))
colors[pred_np==22,0]=255
colors[pred_np==23,2]=255 # handover side
# _pc = pc[0]
# print(pred_np.shape)
# print( torch.mean(_pc[:, pred_np==23], dim=1) )

my_grasp = torch.FloatTensor([0, 0.4, 0])
my_grasp = my_grasp.to(device)

def measure_distance(grasp_point, pred, pc):
    _pc = pc[0]
    pred = pred[0]

    coef = 1
    #print(_pc[:, pred==22].shape[1])
    if _pc[:, pred==22].shape[1] < 500:
        coef = 0

    centroid = torch.mean(_pc[:, pred==23], dim=1)
    distance = coef*torch.cdist(centroid.unsqueeze(0), grasp_point.unsqueeze(0))

    return distance


def semantic_classifier(grasp_point, pc, c=1):
    d = measure_distance(grasp_point, pred, pc)
    d = c*(d-0.03)
    return d





def _velocity(grasp_point, pc, Nq=1, Np=1):
        # eulers [B,3]
        # translations [B,3]
        # pc [B,1024,3] - normalized

        grasp_point_t = grasp_point.clone()
        grasp_point_t.requires_grad_(True)
        if grasp_point_t.grad is not None:
            grasp_point_t.grad.zero_()
            
        pc.requires_grad_(False)
        d_score = semantic_classifier(grasp_point_t, pc)
            
        success = torch.sigmoid(d_score)

        Nq = torch.FloatTensor([Nq]).to(device)
        Np = torch.FloatTensor([Np]).to(device)
        bias_term = torch.log(Nq) - torch.log(Np)
        d_score -= bias_term


        d_score.backward(torch.ones_like(d_score).to(device))
        grad = grasp_point_t.grad
    

        return grad, success

init_grasp_point = my_grasp.clone()
eta = 0.01
for t in range(1, 100):

    # get gradient
    grad, success = _velocity(my_grasp, pc)
    # rots_v, translations_v, success = self._velocity(_rots,_translations, Nq=self.args.Nq, Np=self.args.Np, pc=_pc)

    my_grasp = my_grasp.data - eta*grad

print(my_grasp)



_pc = pc[0].permute(1,0).cpu().numpy()
pc_mesh = trimesh.points.PointCloud(vertices=_pc, colors=colors)
scene = trimesh.Scene()
scene.add_geometry(pc_mesh)
ss1 = trimesh.primitives.Sphere(center=init_grasp_point.detach().cpu().numpy(), radius=0.01)
ss1.visual.face_colors = [255,0,0,255]
ss2 = trimesh.primitives.Sphere(center=my_grasp.detach().cpu().numpy(), radius=0.01)
ss2.visual.face_colors = [0,0,255,255]
# ss2 = trimesh.primitives.Sphere(center=my_grasp.detach().cpu().numpy())
scene.add_geometry(ss1)
scene.add_geometry(ss2)

scene.show()



