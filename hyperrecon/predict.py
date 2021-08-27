"""
Test/inference functions for HyperRecon
For more details, please read:
  Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
  "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 

"""
from glob import glob
import os

from hyperrecon.util.train import BaseTrain

from hyperrecon.util import utils

class Predict(BaseTrain):
  '''Predict.
  
  Kind of hacky. Overrides the BaseTrain class,
  doesn't perform any training steps but simply
  loads from the most recent model and performs
  evaluation.
  '''
  def __init__(self, args):
    super(Predict, self).__init__(args=args)
    model_paths = sorted(glob(os.path.join(self.ckpt_dir, '*')))
    if len(model_paths) == 0:
      raise ValueError('No models found for prediction.')
    else:
      self.load = model_paths[-1]

  def train(self):
    print('Predict')
    self.train_begin()
    self.train_epoch_end(is_eval=True, is_save=False)
    self.train_end(verbose=True)
    

# def get_everything(path, device, \
#            cp=None, n_grid=20, \
#            gt_data=None, xdata=None, seg_data=None, 
#            normalized=True, split='test', legacy_dataset=False,
#            num_subjects=10, give_recons=False):
  
#   # Forward through latest available model
#   if cp is None:
#     glob_path = path.replace('[', '[()').replace(']', '()]').replace('()', '[]')
#     model_paths = sorted(glob.glob(os.path.join(glob_path, 'checkpoints/model.*.h5')))
#     model_path = model_paths[-1]
#   # Or forward through specified epoch
#   else:
#     model_path = os.path.join(path, 'checkpoints/model.{epoch:04d}.h5'.format(epoch=cp))
    
#   args_txtfile = os.path.join(path, 'args.txt')
#   if os.path.exists(args_txtfile):
#     with open(args_txtfile) as json_file:
#       args = json.load(json_file)
#   else:
#     raise Exception('no args found')
#   args['normalized'] = normalized

#   if args['legacy_dataset']:
#     N = 32
#     xdata = dataset.get_test_data(maskname='16p3')[:N]
#     gt_data = dataset.get_test_gt()[:N]
#     testset = dataset.Dataset(xdata, gt_data)
#     params = {'batch_size': args['batch_size'],
#        'shuffle': False,
#        'num_workers': 0, 
#        'pin_memory': True}
#     out_shape = xdata.shape

#   else:
#     transform = transforms.Compose([layers.ClipByPercentile()])
#     testset = dataset.SliceDataset(args['data_path'], 'test', total_subjects=num_subjects, 
#                   transform=transform, include_seg=True)
#     params = {'batch_size': 192,
#        'shuffle': False,
#        'num_workers': 0, 
#        'pin_memory': True}
#     out_shape = [len(testset), 160, 224, 1]

#   dataloader = torch.utils.data.DataLoader(testset, **params)

#   losses = args['losses']
#   range_restrict = args['range_restrict']
#   topK = args['topK']
#   hyperparameters = args['hyperparameters']
#   maskname = args['undersampling_rate']

#   num_hyperparams = len(losses)-1 if range_restrict else len(losses)

#   if hyperparameters is not None:
#     hps = torch.tensor([hyperparameters]).unsqueeze(1).float().to(device)
#   elif len(losses) == 3:
#     alphas = np.linspace(0, 1, n_grid)
#     betas = np.linspace(0, 1, n_grid)
#     hps = torch.tensor(np.stack(np.meshgrid(alphas, betas), -1).reshape(-1,2)).float().to(device)
#     if not range_restrict:
#       hps = utils.oldloss2newloss(hps)
#   elif len(losses) == 2:
#     hps = torch.linspace(0, 1, n_grid).view(-1, 1).float().to(device)
#     if not range_restrict:
#       hps = utils.oldloss2newloss(hps)


#   args['mask'] = dataset.get_mask('160_224', maskname).to(device)
#   args['device'] = device
#   n_ch_in = 2

#   if args['hyperparameters'] is None:
#     network = model.HyperUnet(
#              num_hyperparams,
#              args['hnet_hdim'],
#              hnet_norm=not args['range_restrict'],
#              in_ch_main=n_ch_in,
#              out_ch_main=args['n_ch_out'],
#              h_ch_main=args['unet_hdim'], 
#              ).to(args['device'])
#   else:
#     network = model.Unet(in_ch=n_ch_in,
#                out_ch=args['n_ch_out'], 
#                h_ch=args['unet_hdim']).to(args['device'])
#   # print('Total parameters:', utils.count_parameters(network))

#   network = utils.load_checkpoint(network, model_path)
#   criterion = losslayer.AmortizedLoss(losses, range_restrict, args['sampling'], topK, device, args['mask'], take_avg=False)

#   gr = give_recons
#   gl = True
#   return test(network, dataloader, args, hps, args['normalized'], out_shape, criterion=criterion, give_recons=gr, give_losses=gl)

# def test(network, dataloader, args, hps, normalized, out_shape, criterion=None, \
#     give_recons=False, give_losses=False):
#   """Testing for a fixed set of hyperparameter setting.

#   Returns recons, losses, and metrics (if specified)
#   For every sample in the dataloader, evaluates with all hyperparameters in hps.
#   Batch size must match size of dataset (TODO change this)

#   If take_avg is True, then returns [len(hps)]
#   """
#   network.eval()

#   res = {}
#   if give_recons:
#     res['recons'] = np.full((len(hps), *out_shape), np.nan)
#     res['gts'] = np.full((len(hps), *out_shape), np.nan)
#     res['gt_segs'] = np.full((len(hps), *out_shape), np.nan)
#     res['recon_segs'] = np.full((len(hps), *out_shape), np.nan)
#     res['pred_gt_segs'] = np.full((len(hps), *out_shape), np.nan)
#   if give_losses:
#     res['losses'] = np.full((len(hps), out_shape[0]), np.nan)
#     # res['dcs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     # res['cap_regs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     # res['ws'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     # res['tvs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     res['psnrs'] = np.full((len(hps), out_shape[0]), np.nan)
#     # all_rpsnrs = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     res['mses'] = np.full((len(hps), out_shape[0]), np.nan)
#     res['ssims'] = np.full((len(hps), out_shape[0]), np.nan)
#     res['l1s'] = np.full((len(hps), out_shape[0]), np.nan)
#     # res['percs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     res['dices'] = np.full((len(hps), out_shape[0]), np.nan)
#     res['dices_gt'] = np.full((len(hps), out_shape[0]), np.nan)

#   for h, hp in enumerate(hps):
#     print(hp)
#     for i, batch in enumerate(dataloader): 
#       zf, gt, y, seg = utils.prepare_batch(batch, args, split='test')
#       batch_size = len(zf)

#       hyperparams = hp.expand(batch_size, -1)
#       with torch.set_grad_enabled(False):

#         if args['hyperparameters'] is None: # Hypernet
#           preds = network(zf, hyperparams)
#         else:
#           preds = network(zf) # Baselines
#         loss = criterion(preds, y, hyperparams, None, target=gt)


#       if give_losses:
#         assert criterion is not None, 'loss must be provided'
#         res['losses'][h, i*batch_size:i*batch_size+len(preds)] = loss.cpu().detach().numpy()
#         # dcs.append(regs['dc'].cpu().detach().numpy())
#         # cap_regs.append(regs['cap'].cpu().detach().numpy())
#         # tvs.append(regs['tv'].cpu().detach().numpy())
#         res['psnrs'][h, i*batch_size:i*batch_size+len(preds)] = utils.get_metrics(gt, preds, zf, metric_type='psnr', normalized=normalized)
#         # rpsnrs = utils.get_metrics(gt, preds, zf, metric_type='relative psnr', normalized=normalized)
#         res['mses'][h, i*batch_size:i*batch_size+len(preds)] = criterion.get_mse(gt, preds).detach().cpu().numpy()
#         res['ssims'][h, i*batch_size:i*batch_size+len(preds)] = criterion.get_ssim(gt, preds).detach().cpu().numpy()
#         res['l1s'][h, i*batch_size:i*batch_size+len(preds)] = criterion.get_l1(gt, preds).detach().cpu().numpy()
#         # percs = criterion.get_watson_dft(gt, preds).detach().cpu().numpy()
#         dices, dices_gt, pred_segs, pred_gt_segs, target_segs = criterion.get_dice(preds, gt, seg)#.detach().cpu().numpy()
#         res['dices'][h, i*batch_size:i*batch_size+len(preds)] = dices
#         res['dices_gt'][h, i*batch_size:i*batch_size+len(preds)] = dices_gt

#       if give_recons:
#         res['recons'][h, i*batch_size:i*batch_size+len(preds)] = preds.cpu().detach().numpy()
#         res['gts'][h, i*batch_size:i*batch_size+len(preds)] = gt.cpu().detach().numpy()
#         res['recon_segs'][h, i*batch_size:i*batch_size+len(preds)] = pred_segs#.cpu().detach().numpy()
#         res['gt_segs'][h, i*batch_size:i*batch_size+len(preds)] = target_segs
#         res['pred_gt_segs'][h, i*batch_size:i*batch_size+len(preds)] = pred_gt_segs#.permute(0, 2, 3, 1).cpu().detach().numpy()


#   if give_recons:
#     assert np.isnan(np.sum(res['recons'])) == False, 'Missed some predictions'
#     assert np.isnan(np.sum(res['gts'])) == False, 'Missed some gts'
#   if give_losses:
#     assert np.isnan(np.sum(res['losses'])) == False, 'Missed some predictions'
#     assert np.isnan(np.sum(res['psnrs'])) == False, 'Missed some gts'
#     assert np.isnan(np.sum(res['mses'])) == False, 'Missed some gts'
#     assert np.isnan(np.sum(res['ssims'])) == False, 'Missed some gts'
#     assert np.isnan(np.sum(res['l1s'])) == False, 'Missed some gts'
#     assert np.isnan(np.sum(res['dices'])) == False, 'Missed some gts'

#   return res
