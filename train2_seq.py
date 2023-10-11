import argparse
import json
import os, sys
import csv

from tqdm import tqdm
import pandas as pd

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
torch.backends.cudnn.benchmark = True
from scheduler import CyclicCosineDecayLR

from config_seq import GlobalConfig
from model_efnet_gpt import TransFuser4
#from data2_seq import Mattia_Data

import torchvision

kw='final_'# keyword for the pretrained model in finetune
# data_root = './MultiModeBeamforming/'#path to the dataset

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='train1', help='Unique experiment identifier.') # BEFORE IT WAS test
parser.add_argument('--device', type=str, default='cuda', help='Device to use')							# MAKE SURE TO USE CUDA
parser.add_argument('--epochs', type=int, default=30, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=26, help='Batch size')	# default=24
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')	# /ibex/scratch/tiany0c/log
parser.add_argument('--add_velocity', type = int, default=0, help='concatenate velocity map with angle map')
parser.add_argument('--add_mask', type=int, default=0, help='add mask to the camera data')
parser.add_argument('--enhanced', type=int, default=1, help='use enhanced camera data')
parser.add_argument('--filtered', type=int, default=0, help='use filtered lidar data')
parser.add_argument('--loss', type=str, default='focal', help='crossentropy or focal loss')
parser.add_argument('--scheduler', type=int, default=1, help='use scheduler to control the learning rate')
parser.add_argument('--load_previous_best', type=int, default=0, help='load previous best pretrained model ')
parser.add_argument('--temp_coef', type=int, default=1, help='apply temperature coefficience on the target')
parser.add_argument('--train_adapt_together', type=int, default=1, help='combine train and adaptation dataset together')
parser.add_argument('--finetune', type=int, default=0, help='first train on development set and finetune on 31-34 set')
parser.add_argument('--Test', type=int, default=0, help='Test')
parser.add_argument('--augmentation', type=int, default=0, help='data augmentation of camera and lidar')
parser.add_argument('--angle_norm', type=int, default=1, help='normlize the gps loc with unit, angle can be obtained')
parser.add_argument('--custom_FoV_lidar', type=int, default=0, help='Custom FoV of lidar')
parser.add_argument('--add_seg', type=int, default=0, help='add segmentation on 31&32 images')
parser.add_argument('--ema', type=int, default=1, help='exponential moving average')
parser.add_argument('--flip', type=int, default=0, help='flip all the data to augmentation')
args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)

writer = SummaryWriter(log_dir=args.logdir)
class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""
	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.APL = [] # CHANGE TO APL
		self.bestval = 0
		if args.finetune:
			self.APLft = [0] # APLft
		if args.loss == 'ce':#crossentropy loss
			self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
		elif args.loss == 'focal':#focal loss
			self.criterion = FocalLoss()

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		model.train()
		running_acc = 0.0
		gt_beam_all = []
		pred_beam_all = []
		gt_pwr_all = []
		pred_pwr_all = []
		# Train loop
		pbar=tqdm(dataloader_train, desc='training') # HERE PUT THE DATALOADER BATCH
		for data in pbar:
			# efficiently zero gradients
			optimizer.zero_grad(set_to_none=True)
			# create batch and move to GPU
			gps = data['gps'].to(args.device, dtype=torch.float32)
			image_front_list = []
			image_back_list = []
			for i in range(config.seq_len):
				image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32)) # IF .float() IS NOT INSERTED IT GIVES AN ERROR DUE TO uint8 IMAGE CASTING
				image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))
   
			pred_beams = model(image_front_list + image_back_list, [gps[:,:,0], gps[:,:,1]])
			pred_beam = torch.argmax(pred_beams, dim=1)
			gt_beamidx = data['beamidx'][0].to(args.device, dtype=torch.long)
			gt_beams = data['beam'][0].to(args.device, dtype=torch.float32)
			# running_acc = (pred_beam == gt_beamidx).sum().item()
			if args.temp_coef:#temperature coefficiece
				loss = self.criterion(pred_beams, gt_beams)
			else:
				loss = self.criterion(pred_beams, gt_beamidx)
			gt_beam_all.append(data['beamidx'][0])
			pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
			true_pwr_batch = data['beam_pwr'][0].to(args.device, dtype=torch.float32)
			gt_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx]).cpu().numpy())
			pred_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy()) # HOW TO FIND BATCH SIZE? args.batch_size or pred_beam.shape[0]?
			loss.backward()
			loss_epoch += float(loss.item())
			pbar.set_description(str(loss.item()))
			num_batches += 1
			optimizer.step()
   
			# #============== DEBUG ========================= 
			# estimated_power = (true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy()
			# true_power = (true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx]).cpu().numpy()
			# APL_score = APL(true_power,estimated_power)
			# true_beam = data['beamidx'][0].cpu().numpy()
			# # batch_acc = compute_acc(pred_beam,gt_beamidx,top_k=[1])
			# batch_acc = running_acc / float(pred_beam.shape[0])
			# print(f'\nAPL score = {APL_score:.4f} dB | acc: {batch_acc} | pred vs true beam [{pred_beam.cpu().numpy(), true_beam}]')
			# print(f'power: pred = {estimated_power}, true = {true_power}')
			# a += 1
			# if a == 3:
			# 	break
			#==============================================
			if args.ema:# Exponential Moving Averages
				ema.update()	# during training, after update parameters, update shadow weights

			self.cur_iter += 1
		pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
		gt_beam_all = np.squeeze(np.concatenate(gt_beam_all, 0))
		pred_pwr_all = np.squeeze(np.concatenate(pred_pwr_all, 0)) # REQUIRED TO COMPUTE APL
		gt_pwr_all = np.squeeze(np.concatenate(gt_pwr_all, 0)) # REQUIRED TO COMPUTE APL
		curr_acc = compute_acc(pred_beam_all, gt_beam_all, top_k=[1, 2, 3])
		# print(f'APL score PREDICTED pwr {pred_pwr_all}')
		# print(f'APL score TRUE pwr {gt_pwr_all}')
		APL_score = APL(gt_pwr_all, pred_pwr_all) # CALCULATE APL SCORE
		print('Train top beam acc: ',curr_acc, ' APL score: ',APL_score)
		# train_acc = np.sum((pred_beam_all == gt_beam_all))/float(pred_beam_all.shape[0]) ################## DEBUG
		# print(f'Mattia acc: {train_acc:.2f}') ################## DEBUG
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1
		writer.add_scalar('APL_score_train', APL_score, self.cur_epoch)
		for i in range(len(curr_acc)):
			writer.add_scalars('curr_acc_train', {'beam' + str(i):curr_acc[i]}, self.cur_epoch)
		writer.add_scalar('curr_loss_train', loss_epoch, self.cur_epoch)
		if args.finetune:
			if APL_score > self.APLft[-1]:
				self.APLft.append(APL_score)
				print(APL_score, self.APLft[-2], 'save new model')
				torch.save(model.state_dict(), os.path.join(args.logdir, 'all_finetune_on_' + kw + 'model.pth'))
				torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'all_finetune_on_' + kw + 'optim.pth'))
			else:
				print('best',self.APLft[-1])

	def validate(self):
		if args.ema:#Exponential Moving Averages
			ema.apply_shadow()    # before eval\uff0capply shadow weights
		model.eval()
		running_acc = 0.0
		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.
			gt_beam_all=[]
			pred_beam_all=[]
			scenario_all = []
			gt_pwr_all = [] # added co compute APL score
			pred_pwr_all = []
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val), 0):
				# create batch and move to GPU
				gps = data['gps'].to(args.device, dtype=torch.float32)
				image_front_list = []
				image_back_list = []
				for i in range(config.seq_len):
					image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32)) # IF .float() IS NOT INSERTED IT GIVES AN ERROR DUE TO uint8 IMAGE CASTING
					image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))
				
				pred_beams = model(image_front_list + image_back_list, [gps[:,:,0], gps[:,:,1]])
				pred_beam = torch.argmax(pred_beams, dim=1)
				gt_beam_all.append(data['beamidx'][0])
				gt_beams = data['beam'][0].to(args.device, dtype=torch.float32)
				gt_beamidx = data['beamidx'][0].to(args.device, dtype=torch.long)
				pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
				running_acc += (pred_beam == gt_beamidx).sum().item()
				if args.temp_coef:
					loss = self.criterion(pred_beams, gt_beams)
				else:
					loss = self.criterion(pred_beams, gt_beamidx)
				wp_epoch += float(loss.item())
				num_batches += 1
				true_pwr_batch = data['beam_pwr'][0].to(args.device, dtype=torch.float32)
				gt_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx]).cpu().numpy())
				pred_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy())
				scenario_all.append(data['scenario'])
   
				#============== DEBUG ========================= 
				# estimated_power = (true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy()
				# true_power = (true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx]).cpu().numpy()
				# APL_score = APL(estimated_power,true_power)
				# true_beam = data['beamidx'][0].cpu().numpy()
				# print(f'\nAPL score = {APL_score:.4f} dB | acc: {running_acc} | pred vs true beam [{pred_beam.cpu().numpy(), true_beam}]')
				# print(f'power: pred = {estimated_power}, true = {true_power}')
				# if batch_num == 2:
				# 	break
				#============================================== 
    
			pred_beam_all=np.squeeze(np.concatenate(pred_beam_all,0))
			gt_beam_all=np.squeeze(np.concatenate(gt_beam_all,0))
			scenario_all = np.squeeze(np.concatenate(scenario_all,0))
			#calculate accuracy and APL score according to different scenarios
			# scenarios = ['scenario31', 'scenario32', 'scenario33', 'scenario34'] # HOW TO CALCUALTE APL FOR DIFFERENT SCENARIOS??
			# scenarios = ['scenario36']
			# for s in scenarios:
			# 	beam_scenario_index = np.array(scenario_all) == s
			# 	if np.sum(beam_scenario_index) > 0:
			# 		curr_acc_s = compute_acc(pred_beam_all[beam_scenario_index], gt_beam_all[beam_scenario_index], top_k=[1,2,3])
			# 		APL_score_s = compute_APL_score(pred_beam_all[beam_scenario_index], gt_beam_all[beam_scenario_index], max_k=3, delta=5)
			# 		print(s, ' curr_acc: ', curr_acc_s, ' APL_score: ', APL_score_s)
			# 		for i in range(len(curr_acc_s)):
			# 			writer.add_scalars('curr_acc_val', {s + 'beam' + str(i):curr_acc_s[i]}, self.cur_epoch)
			# 		writer.add_scalars('APL_score_val', {s:APL_score_s}, self.cur_epoch)

			curr_acc = compute_acc(pred_beam_all, gt_beam_all, top_k=[1,2,3])
   
			pred_pwr_all = np.squeeze(np.concatenate(pred_pwr_all, 0)) # (n_samples,1)
			gt_pwr_all = np.squeeze(np.concatenate(gt_pwr_all, 0)) # (n_samples,1)
			#============== DEBUG ========================= 
			# print(f'APL score PREDICTED pwr {pred_pwr_all}')
			# print(f'APL score TRUE pwr {gt_pwr_all}')
			# ==============================================
			APL_score_val = APL(gt_pwr_all, pred_pwr_all) # CALCULATE APL SCORE
			wp_loss = wp_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')
			print('Val top beam acc: ',curr_acc, 'APL score: ', APL_score_val)
			writer.add_scalars('APL_score_val', {'scenario_all':APL_score_val}, self.cur_epoch)
			writer.add_scalar('curr_loss_val', wp_loss, self.cur_epoch)

			self.val_loss.append(wp_loss)
			self.APL.append(APL_score_val)

		if args.ema:#Exponential Moving Averages
			ema.restore()	# after eval, restore model parameter


	def test(self):
		model.eval()
		with torch.no_grad():
			pred_beam_all=[]
			pred_beam_confidence = []
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_test), 0): # CHANGE TO dataloader_test
				# create batch and move to GPU
				gps = data['gps'].to(args.device, dtype=torch.float32)
				image_front_list = []
				image_back_list = []
				for i in range(config.seq_len):
					image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32))
					image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))
				
				pred_beams = model(image_front_list + image_back_list, [gps[:,:,0], gps[:,:,1]])
				pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
				sm=torch.nn.Softmax(dim=1)
				beam_confidence=torch.max(sm(pred_beams), dim=1)
				pred_beam_confidence.append(beam_confidence[0].cpu().numpy())

			pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
			pred_beam_confidence = np.squeeze(np.concatenate(pred_beam_confidence, 0))
			save_pred_to_csv(pred_beam_all, top_k=[1, 2, 3], target_csv='beam_pred.csv')
			df = pd.DataFrame(data=pred_beam_confidence)
			df.to_csv('beam_pred_confidence_seq.csv')

	def save(self):
		save_best = False
		print('best', self.bestval, self.bestval_epoch)

		if self.APL[-1] <= self.bestval:
			self.bestval = self.APL[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True

		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'iter': self.cur_iter,
			'bestval': self.bestval,
			'bestval_epoch': self.bestval_epoch,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
			'APL': self.APL,
		}

		# Save ckpt for every epoch
		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'final_model.pth'))
		# # Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))


		if save_best:# save the bestpretrained model
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')
		elif args.load_previous_best:
			model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model.pth')))
			optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'best_optim.pth')))
			tqdm.write('====== Load the previous best model ======>')

class FocalLoss(nn.Module):
	def __init__(self, gamma=2, alpha=0.25):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
	def __call__(self, input, target):
		if len(target.shape) == 1:
			target = torch.nn.functional.one_hot(target, num_classes=64)
		loss = torchvision.ops.sigmoid_focal_loss(input, target.float(), alpha=self.alpha, gamma=self.gamma,
												  reduction='mean')
		return loss

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



def save_pred_to_csv(y_pred, top_k=[1, 2, 3], target_csv='beam_pred.csv'):
	"""
    Saves the predicted beam results to a csv file.
    Expects y_pred: n_samples x N_BEAMS, and saves the top_k columns only.
    """
	cols = [f'top-{i} beam' for i in top_k]
	df = pd.DataFrame(data=y_pred[:, np.array(top_k) - 1]+1, columns=cols)
	df.index.name = 'index'
	df.to_csv(target_csv)
def compute_acc(y_pred, y_true, top_k=[1,2,3]):
    """ Computes top-k accuracy given prediction and ground truth labels."""
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)
    n_test_samples = len(y_true)
    if len(y_pred) != n_test_samples:
        raise Exception('Number of predicted beams does not match number of labels.')
    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(y_true)):
        for k_idx in range(n_top_k):
            hit = np.any(y_pred[samp_idx,:top_k[k_idx]] == y_true[samp_idx])
            total_hits[k_idx] += 1 if hit else 0
    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(y_true)*100, 4)


def APL(true_best_pwr, est_best_pwr):
    """
    Average Power Loss: average of the power wasted by using the predicted beam
    instead of the ground truth optimum beam.
    """
    
    return np.mean(10 * np.log10(est_best_pwr / true_best_pwr))


# def dataset_augmentation(root_csv):

# 	# return augmentation on input dataset	

# 	# camera augmentation: total 7
# 	# lidar augmentation: total 2
# 	# radar augmentation: total 1
# 	# return: ((camera_aug_num + 1) * (lidar_aug_num + 1) * (radar_aug_num + 1)) - 1

# 	camera_aug_num = 7
# 	lidar_aug_num = 2
# 	radar_aug_num = 1
# 	augmentation_set = []
# 	for i in range(0, camera_aug_num + 1):
# 		for j in range(0, lidar_aug_num + 1):
# 			for k in range(0, radar_aug_num + 1):
# 				if i == 0 and j == 0 and k == 0:	# skip the original dataset
# 					continue
# 				augmentation_entry = CARLA_Data(root=val_root, root_csv=root_csv, config=config, test=False, augment={'camera':i, 'lidar':j, 'radar':k})
# 				if augmentation_set == []:
# 					augmentation_set = augmentation_entry
# 				else:
# 					augmentation_set = ConcatDataset([augmentation_set, augmentation_entry])
# 	print('Augmented Dataset: ', root_csv, ' Samples: ', str(len(augmentation_set)))
# 	return augmentation_set


# Config
config = GlobalConfig()
config.add_velocity = args.add_velocity
config.add_mask = args.add_mask
config.enhanced = args.enhanced
config.angle_norm = args.angle_norm
config.custom_FoV_lidar=args.custom_FoV_lidar
config.filtered = args.filtered
config.add_seg = args.add_seg
data_root = config.data_root	# path to the dataset

import random
import numpy
seed = 100
random.seed(seed)
np.random.seed(seed) # numpy
torch.manual_seed(seed) # torch+CPU
# torch.cuda.manual_seed(seed) # torch+GPU
torch.use_deterministic_algorithms(False)
g = torch.Generator()
g.manual_seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# ============ DATASET
# from mattiamodel2_seq import TransFuser
from data2_seq import MATTIA_Data

scenario36_root_csv = './scenario36/scenario36.p'
root_csv_train = './deepsense_challenge2023_trainset.csv'
dataset_scen36 = MATTIA_Data(root_scenario_csv=scenario36_root_csv,root_csv=root_csv_train,num_scenario=36,config=config)

train_test_split = 0.9 # PUT THESE HYPERPARAMETERS INTO CONFIG() FILE OR args(), maybe better in args()
train_val_split = 0.8

train_dim = int(train_test_split * len(dataset_scen36))
train_dim1 = int(train_val_split * train_dim)
train_set, test_set = torch.utils.data.random_split(dataset_scen36,[train_dim, len(dataset_scen36) - train_dim])
train_set, val_set = torch.utils.data.random_split(train_set,[train_dim1,len(train_set) - train_dim1])
train_train_set, val_set = torch.utils.data.random_split(dataset_scen36,[train_dim,len(dataset_scen36) - train_dim])
dataloader_train = DataLoader(train_set,batch_size=args.batch_size,shuffle=True, num_workers=8, pin_memory=True,
								  worker_init_fn=seed_worker, generator=g)
dataloader_val = DataLoader(val_set,batch_size=args.batch_size,shuffle=True, num_workers=8, pin_memory=False,
								  worker_init_fn=seed_worker, generator=g)
dataloader_test = DataLoader(test_set,batch_size=args.batch_size,shuffle=False, num_workers=8, pin_memory=False,
								  worker_init_fn=seed_worker, generator=g)
#train_set, val_set = torch.utils.data.random_split(dataset_scen36,[20,len(dataset_scen36) - 20])
#dataloader = DataLoader(train_set,batch_size=args.batch_size,shuffle=False)

# Model
model = TransFuser4(config, args.device)
# model = torch.nn.DataParallel(model)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
if args.scheduler:#Cyclic Cosine Decay Learning Rate
	scheduler = CyclicCosineDecayLR(optimizer,
	                                init_decay_epochs=15,
	                                min_decay_lr=1e-5, # 2.5e-6
	                                restart_interval = 10,
	                                restart_lr= 1e-4, # 12.5e-5
	                                warmup_epochs=10,
	                                warmup_start_lr=1e-5) # 2.5e-6
 
trainer = Engine()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('======Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('======Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	print('======Loading checkpoint from ' + args.logdir)
	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
		log_table = json.load(f)

	# Load variables
	trainer.cur_epoch = log_table['epoch']
	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
	trainer.bestval = log_table['bestval']
	trainer.train_loss = log_table['train_loss']
	trainer.val_loss = log_table['val_loss']
	trainer.APL = log_table['APL']


	# # FOR TESTING ONLY

	# Load checkpoint
	if args.finetune:# finetune the pretrained model

		if os.path.exists(os.path.join(args.logdir, 'all_finetune_on_'+ kw + 'model.pth')):
			print('======loading last'+'all_finetune_on_'+ kw + 'model.pth')
			model.load_state_dict(torch.load(os.path.join(args.logdir, 'all_finetune_on_'+ kw + 'model.pth')))
			optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'all_finetune_on_' + kw + 'optim.pth')))
		else:
			print('======loading '+kw+' model')
			model.load_state_dict(torch.load(os.path.join(args.logdir, kw+'model.pth')))
	else:
		print('======loading best_model')
		# model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model.pth')))


ema = EMA(model, 0.999)

if args.ema:
	ema.register()

# Log args
# with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
# 	json.dump(args.__dict__, f, indent=2)
if args.Test:
	trainer.test()
	print('Test finish')
else:
	for epoch in range(trainer.cur_epoch, args.epochs):
		print('epoch:',epoch)
		trainer.train()
		if not args.finetune:
			trainer.validate()
			trainer.save()
		if args.scheduler:
			print('lr', scheduler.get_lr())
			scheduler.step()
