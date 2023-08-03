import logging
import numpy as np
import torch
from torch.utils.data import Dataset


class NUSDataset(Dataset):

    def __init__(self, mesh_paths, gt_paths, radius, con_num=2, offset=None):
        super().__init__()

        self.flows = []
        if gt_paths is None:
            self.gts = None
        else:
            self.gts = []
        self.skip_list = [0]

        self.len = 0
        self.radius = radius
        self.offset = offset
        self.con_num = con_num
        
        if not offset:
            self.offset = radius

        for i in range(len(mesh_paths)):
            logging.info("Loading Deep MeshFlows: {}".format(mesh_paths[i]))
            if self.gts is not None:
                logging.info("Loading GTs: {}".format(gt_paths[i]))

            flows = np.load(mesh_paths[i])
            flows = flows.transpose(0, 3, 1, 2)
            self.flows.append(flows)
            
            if self.gts is not None:
                gts = np.load(gt_paths[i])
                gts = -1.0 * gts
                gts = gts.transpose(0, 3, 1, 2)
                self.gts.append(gts)

            self.len += flows.shape[0] - 2 * radius - con_num + 2
            self.skip_list.append(self.len)
        
        logging.info("Total Length: {}".format(self.len))


    def __getitem__(self, index):
        # flows (con_num,     2 * 2 * raidus, H, W)
        # gts   (con_num,     2,              H, W)

        pos = 0
        for i in range(len(self.flows)):
            if index >= self.skip_list[i] and index < self.skip_list[i+1]:
                pos = i
                index -= self.skip_list[i]
                break

        flows = None
        for c in range(self.con_num):
            temp = self.flows[pos][index+c]
            for i in range(index+1, index+2*self.radius):
                temp = np.concatenate((temp, self.flows[pos][i+c]), axis=0)

            if c == 0:
                flows = np.expand_dims(temp, axis=0)
            else:
                temp = np.expand_dims(temp, axis=0)
                flows = np.concatenate((flows, temp), axis=0)
        flows = torch.tensor(flows)
        
        gts = None
        if self.gts is not None:
            for c in range(self.con_num):
                gts_temp = self.gts[pos][index+self.offset+c]
                if c == 0:
                    gts = np.expand_dims(gts_temp, axis=0)
                else:
                    gts_temp = np.expand_dims(gts_temp, axis=0)
                    gts = np.concatenate((gts, gts_temp), axis=0)
            gts = torch.tensor(gts)
        else:
            gts = flows[:, 0:2, :, :]

        return flows, gts
    
    def __len__(self):
        return self.len