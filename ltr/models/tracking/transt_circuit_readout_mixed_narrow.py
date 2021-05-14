import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.circuit_featurefusion_network_mixed import build_featurefusion_network
from ltr.admin.loading import load_weights
from ltr.models.neck.circuit_rnn import ConvGRUCell
import ltr.data.processing_utils as prutils


class TransT(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes, input_dim=32):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        # self.circuit = circuit
        hidden_dim = featurefusion_network.d_model
        # self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.rnn_embed = MLP(hidden_dim, hidden_dim * 2, hidden_dim, 3)
        self.class_embed_new = MLP(hidden_dim * 1, hidden_dim, num_classes + 1, 3)
        self.bbox_embed_new = MLP(hidden_dim * 1, hidden_dim, 4, 3)
        # self.class_embed_new = MLP(hidden_dim * 1, hidden_dim, num_classes + 1, 3)
        # self.bbox_embed_new = MLP(hidden_dim * 1, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.rnn_dims = 32
        self.rnn_proj = nn.Conv3d(backbone.num_channels, self.rnn_dims, kernel_size=1)
        self.rnn_decode = nn.Conv2d(self.rnn_dims, hidden_dim, kernel_size=1)
        self.nl = F.softplus
        self.backbone = backbone
        self.reset_hidden = False
        self.circuit = ConvGRUCell(input_dim=self.rnn_dims, hidden_dim=self.rnn_dims, kernel_size=3, padding_mode='zeros', batchnorm=True, use_attention=True, timesteps=8)

    def _generate_label_function(self, target_bb, sigma, kernel, feature, output_sz, end_pad_if_even, target_absent=None):
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), sigma,
                                                      kernel,
                                                      feature, output_sz,
                                                      end_pad_if_even=end_pad_if_even)
        if target_absent is not None:
            gauss_label *= (1 - target_absent).view(-1, 1, 1).float()
        return gauss_label

    def forward(self, search, template, samp_idx, labels, settings):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        # Reshape search into a 4D tensor
        search_shape = [int(x) for x in search.shape]
        search = search.view([search_shape[0] * search_shape[1]] + search_shape[2:])
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        with torch.no_grad():
            feature_search, pos_search = self.backbone(search)
            feature_template, pos_template = self.backbone(template)
        src_search, mask_search = feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None

        # Use the circuit to track through the features
        post_src_search_shape = [int(x) for x in src_search.shape]
        post_mask_search_shape = [int(x) for x in mask_search.shape]
        src_search = src_search.view(search_shape[:2] + post_src_search_shape[1:])
        mask_search = mask_search.view(search_shape[:2] + post_mask_search_shape[1:])
        proc_labels = self._generate_label_function(
                labels.to("cpu"),
                settings.sigma,
                settings.kernel,
                settings.feature,  # post_src_search_shape[-1],  # settings.feature,
                settings.output_sz,  # post_src_search_shape[-1],  # settings.output_sz,
                settings.end_pad_if_even)
        exc, inh = None, None
        rnn_src_search = self.nl(self.rnn_proj(src_search.permute(0, 2, 1, 3, 4)))
        proc_label_shape = [int(x) for x in proc_labels.shape]
        proc_labels = proc_labels.view(search_shape[:2] + [1] + proc_label_shape[1:]).to(src_search.device)  # .mean(2)
        # from matplotlib import pyplot as plt
        # im=0;ti=6;plt.subplot(121);plt.imshow(proc_labels[im, ti].cpu());plt.subplot(122);plt.imshow(src_search[im, ti].mean(0).detach().cpu());plt.show()
        for t in range(samp_idx):
            exc, inh = self.circuit(rnn_src_search[:, :, t], excitation=exc, inhibition=inh, label=proc_labels[:, t])

        # Reshape pos_search
        pos_search_shape = [int(x) for x in pos_search[-1].shape]
        pos_search = pos_search[-1].view(search_shape[:2] + pos_search_shape[1:])

        # Split off a pair of TransT features and then use the circuit to gate the Qs in the decoder.
        src_search = self.input_proj(src_search[:, samp_idx])
        src_template = self.input_proj(src_template)
        mask_search = mask_search[:, samp_idx]
        pos_search = pos_search[:, samp_idx]
        hs, exc = self.featurefusion_network(src_template, mask_template, src_search, mask_search, pos_template[-1], pos_search, exc=self.rnn_decode(exc))

        # Concat exc to hs too
        # hs = torch.cat([hs, self.rnn_embed(exc)], -1)
        outputs_class = self.class_embed_new(hs)
        outputs_coord = self.bbox_embed_new(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def reset_states(self):
        self.reset_hidden = True  # exc = None
        self.exc = None
        self.inh = None
        print("Reset hidden states.")

    def track(self, search, info):
        imshape = search.shape
        ims = search.clone()
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        features_search, pos_search = self.backbone(search)
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search= features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        # Use the circuit to track through the features
        src_search = self.input_proj(src_search)
        if self.reset_hidden:  # not hasattr(self, "exc"):
            exc, inh = None, None
            # proc_labels = self._generate_label_function(torch.Tensor([16, 16, 17, 17]), 0.1, 1, src_search.shape[-1], imshape[-1], False)
            proc_labels = self._generate_label_function(torch.Tensor([16, 16, 17, 17]), 0.05, 4, src_search.shape[-1], imshape[-1], False)
            proc_labels = proc_labels.to(src_search.device)[None]
        else:
            exc = self.exc
            inh = self.inh
            proc_labels = None
        exc, inh = self.circuit(src_search, excitation=exc, inhibition=inh, label=proc_labels, reset_hidden=self.reset_hidden)
        # from matplotlib import pyplot as plt
        # im=0;ti=0;plt.subplot(141);plt.imshow(proc_labels[im].squeeze().cpu());plt.subplot(142);plt.imshow(src_search[im].mean(0).detach().cpu());plt.subplot(143);plt.imshow(exc.squeeze().mean(0).detach().cpu());plt.subplot(144);plt.imshow(ims.squeeze().permute(1, 2, 0).cpu());plt.show()
        self.reset_hidden = False
        self.exc = exc
        self.inh = inh
        # Reshape exc for the transformers
        hs, exc = self.featurefusion_network(self.input_proj(src_template), mask_template, src_search, mask_search, pos_template[-1], pos_search[-1], exc=exc)

        # Concat exc to hs too
        # hs = torch.cat([hs, exc], -1)
        outputs_class = self.class_embed_new(hs)
        outputs_coord = self.bbox_embed_new(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template

class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = target_boxes[:, :4]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def transt_resnet50(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    device = torch.device(settings.device)
    model.to(device)
    if settings.init_ckpt:
        print("Initializing from settings.init_ckpt")
        model = load_weights(model, settings.init_ckpt, strict=False)  # Not strict so we can add to the model
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
