from hyperrecon.util.train import BaseTrain

class UHS(BaseTrain):
  """UHS."""

  def __init__(self, args):
    super(UHS, self).__init__(args=args)

  def set_hparams(self, hparams):
    # Parameters related to model selection.
    self.val_mode = 'max'
    self.val_metric = 'logit.auc'
    self.val_op = np.greater_equal if self.val_mode == 'max' else np.less_equal
    self.val_best = -np.inf if self.val_mode == 'max' else np.inf

    # Algorithm-specific parameter
    self.mixup_beta = hparams.mixup_beta
    self.n_embedding = int(hparams.n_embedding) if isinstance(
        hparams.n_embedding, (int, float)) else 0

    # File suffix
    self.file_suffix = 'beta{}'.format(
        self.mixup_beta) if self.mixup_beta > 0 else ''

  def set_metrics(self):
    # Metrics
    self.list_of_metrics = ['loss.train', 'loss.xe', 'loss.L2', 'acc.train']
    self.list_of_eval_metrics = ['embed.auc', 'pool.auc', 'pool.ap'] + [
        'embed.esc_at_ovk_%g' % r for r in [0.1, 0.05, 0.01, 0.001]
    ] + ['pool.esc_at_ovk_%g' % r for r in [0.1, 0.05, 0.01, 0.001]
        ] + ['pool.rec_at_prec_%g' % r for r in [0.3, 0.5, 0.7, 0.9]]
    self.list_of_eval_metrics += [
        #'embed.kocsvm',
        #'embed.gocsvm',
        #'embed.locsvm',
        #'embed.dsvdd',
        #'embed.kde10',
        #'embed.kde30',
        #'embed.kde100',
        #'embed.gde',
        #'embed.gde2',
        #'embed.gde5',
        #'pool.kocsvm',
        #'pool.gocsvm',
        #'pool.locsvm',
        #'pool.dsvdd',
        #'pool.kde10',
        #'pool.kde30',
        #'pool.kde100',
        #'pool.gde',
        #'pool.gde2',
        #'pool.gde5',
        #'pool.seg_auc',
        #'pool.seg_ap',
        'pool.segloc_auc',
        'pool.segloc_ap',
    ]
    self.list_of_eval_metrics += [
        metric + '.ema' for metric in self.list_of_eval_metrics
    ]
    self.metric_of_interest = [
        #'embed.auc',
        #'embed.esc_at_ovk_0.05',
        #'embed.esc_at_ovk_0.01',
        #'pool.auc',
        #'pool.esc_at_ovk_0.05',
        #'pool.esc_at_ovk_0.01',
        #'logit.auc',
        #'logit.esc_at_ovk_0.05',
        #'logit.esc_at_ovk_0.01',
        #'dscore.auc',
        #'dscore.esc_at_ovk_0.05',
        #'dscore.esc_at_ovk_0.01',
        #'embed.kocsvm',
        #'embed.gocsvm',
        #'embed.locsvm',
        #'embed.dsvdd',
        #'embed.kde10',
        #'embed.kde30',
        #'embed.kde100',
        #'embed.gde',
        #'embed.gde2',
        #'embed.gde5',
        #'pool.kocsvm',
        #'pool.gocsvm',
        #'pool.locsvm',
        #'pool.dsvdd',
        #'pool.kde10',
        #'pool.kde30',
        #'pool.kde100',
        #'pool.gde',
        #'pool.ap',
        #'pool.rec_at_prec_0.3',
        #'pool.rec_at_prec_0.5',
        #'pool.rec_at_prec_0.7',
        #'pool.rec_at_prec_0.9',
        #'pool.gde2',
        #'pool.gde5',
        #'pool.seg_auc',
        #'pool.seg_ap',
        'pool.segloc_auc',
        'pool.segloc_ap',
    ]
    self.metric_of_interest += [
        metric + '.ema' for metric in self.metric_of_interest
    ]
    assert all([
        m in self.list_of_eval_metrics for m in self.metric_of_interest
    ]), 'Some metric does not exist'
