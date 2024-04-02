import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import BCEWithLogitsLoss
from torchmetrics import Precision, Recall, F1

# import torch.nn.functional as F
#
# # from timm.models.layers import trunc_normal_
#
# from src.benchmark.transfer_change_detection.models.segmenter_utils_new import padding, unpadding
#
#
import src.benchmark.transfer_change_detection.models.torch_utils_new as ptu
from src.benchmark.transfer_change_detection.model.factory import create_segmenter


#
#
# class Segmenter(nn.Module):
#     def __init__(
#             self,
#             encoder,
#             decoder,
#             n_cls,
#     ):
#         super().__init__()
#         self.n_cls = n_cls
#         self.patch_size = encoder.patch_size
#         self.encoder = encoder
#         self.decoder = decoder
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         def append_prefix_no_weight_decay(prefix, module):
#             return set(map(lambda x: prefix + x, module.no_weight_decay()))
#
#         nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
#             append_prefix_no_weight_decay("decoder.", self.decoder)
#         )
#         return nwd_params
#
#     def forward(self, im):
#         H_ori, W_ori = im.size(2), im.size(3)
#         im = padding(im, self.patch_size)
#         H, W = im.size(2), im.size(3)
#
#         x = self.encoder(im, return_features=True)
#
#         # remove CLS/DIST tokens for decoding
#         num_extra_tokens = 1 + self.encoder.distilled
#         x = x[:, num_extra_tokens:]
#
#         masks = self.decoder(x, (H, W))
#
#         masks = F.interpolate(masks, size=(H, W), mode="bilinear")
#         masks = unpadding(masks, (H_ori, W_ori))
#
#         return masks
#
#     def get_attention_map_enc(self, im, layer_id):
#         return self.encoder.get_attention_map(im, layer_id)
#
#     def get_attention_map_dec(self, im, layer_id):
#         x = self.encoder(im, return_features=True)
#
#         # remove CLS/DIST tokens for decoding
#         num_extra_tokens = 1 + self.encoder.distilled
#         x = x[:, num_extra_tokens:]
#
#         return self.decoder.get_attention_map(x, layer_id)


def dice_loss(out,mask,epsilon=1e-5):
    inter = torch.dot(out.reshape(-1), mask.reshape(-1))
    sets_sum = torch.sum(out) + torch.sum(mask)
    return (2 * inter + epsilon) / (sets_sum + epsilon)


class ChangeDetectionModel(LightningModule):
    def __init__(self, backbone_cp, lr):
        super().__init__()
        self.lr = lr
        backbone_kwargs = {
            'type': 'VisionTransformer', 'image_size': [224, 224], 'patch_size': 16, 'in_channels': 13, 'embed_dims': 384,
            'num_layers': 12, 'num_heads': 6, 'drop_path_rate': 0.1, 'attn_drop_rate': 0.0, 'drop_rate': 0.0,
            'final_norm': True, 'norm_cfg': {'type': 'LN', 'eps': 1e-06, 'requires_grad': True}, 'with_cls_token': True,
            'interpolate_mode': 'bicubic', 'checkpoint': backbone_cp
        }
        decode_kwargs = {
            'type': 'SegmenterMaskTransformerHead', 'in_channels': 384, 'channels': 384, 'num_classes': 100,
            'num_layers': 2, 'num_heads': 6, 'embed_dims': 384, 'dropout_ratio': 0.0,
            'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}
        }
        net_kwargs = {"backbone": backbone_kwargs, "decoder": decode_kwargs}

        self.model = create_segmenter(net_kwargs)
        self.model.to(ptu.device)
        self.first_conv = nn.Conv2d(100, 1, 1, stride=1)
        # self.second_conv = nn.Conv2d(10, 1, 1, stride=1)

        self.criterion = BCEWithLogitsLoss()
        self.dice_loss = dice_loss
        self.prec = Precision(multiclass=False,threshold=0.5)
        self.rec = Recall(multiclass=False,threshold=0.5)
        self.f1 = F1(multiclass=False,threshold=0.5)


    def forward(self, x1, x2):
        x1_out = self.model(x1)
        x2_out = self.model(x2)
        x_diff = x1_out - x2_out

        # x_comb = torch.concat([x1_out, x2_out], 1)
        second_conv_out = self.first_conv(x_diff)
        # second_conv_out = self.second_conv(first_conv_out)
        return second_conv_out

    def training_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True,sync_dist=True)
        self.log('train/precision', prec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('train/recall', rec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        #if args.nc == 3:
        #    tensorboard.add_image('train/img_1', img_1[0], global_step)
        #    tensorboard.add_image('train/img_2', img_2[0], global_step)
        #    tensorboard.add_image('train/mask', mask[0], global_step)
        #    tensorboard.add_image('train/out', pred[0], global_step)
        #else:
        #    tensorboard.add_image('train/img_1', img_1[0,1:4,:,:], global_step)
        #    tensorboard.add_image('train/img_2', img_2[0,1:4,:,:], global_step)
        #    tensorboard.add_image('train/mask', mask[0], global_step)
        #    tensorboard.add_image('train/out', pred[0], global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True,sync_dist=True)
        self.log('val/precision', prec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('val/recall', rec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        #if args.nc == 3:
        #    tensorboard.add_image('val/img_1', img_1[0], global_step)
        #    tensorboard.add_image('val/img_2', img_2[0], global_step)
        #    tensorboard.add_image('val/mask', mask[0], global_step)
        #    tensorboard.add_image('val/out', pred[0], global_step)
        #else:
        #    tensorboard.add_image('val/img_1', img_1[0,1:4,:,:], global_step)
        #    tensorboard.add_image('val/img_2', img_2[0,1:4,:,:], global_step)
        #    tensorboard.add_image('val/mask', mask[0], global_step)
        #    tensorboard.add_image('val/out', pred[0], global_step)
        return loss

    def shared_step(self, batch):
        img_1, img_2, mask = batch
        out = self(img_1, img_2)
        pred = torch.sigmoid(out)
        loss = self.criterion(out, mask)
        # + self.dice_loss(out,mask)
        prec = self.prec(pred, mask.long())
        rec = self.rec(pred, mask.long())
        f1 = self.f1(pred, mask.long())
        return img_1, img_2, mask, pred, loss, prec, rec, f1

    def configure_optimizers(self):
        # params = self.model.parameters()
        params = set(self.model.parameters()).difference(self.model.encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]

#
#
# def vit_base_patch8_384(pretrained=False, **kwargs):
#     """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer(
#         "vit_base_patch8_384",
#         pretrained=pretrained,
#         default_cfg=dict(
#             url="",
#             input_size=(3, 384, 384),
#             mean=(0.5, 0.5, 0.5),
#             std=(0.5, 0.5, 0.5),
#             num_classes=1000,
#         ),
#         **model_kwargs,
#     )
#     return model
#
#
# def create_vit(model_cfg):
#     model_cfg = model_cfg.copy()
#     backbone = model_cfg.pop("backbone")
#
#     normalization = model_cfg.pop("normalization")
#     model_cfg["n_cls"] = 1000
#     mlp_expansion_ratio = 4
#     model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]
#
#     if backbone in default_cfgs:
#         default_cfg = default_cfgs[backbone]
#     else:
#         default_cfg = dict(
#             pretrained=False,
#             num_classes=1000,
#             drop_rate=0.0,
#             drop_path_rate=0.0,
#             drop_block_rate=None,
#         )
#
#     default_cfg["input_size"] = (
#         3,
#         model_cfg["image_size"][0],
#         model_cfg["image_size"][1],
#     )
#     model = VisionTransformer(**model_cfg)
#     if backbone == "vit_base_patch8_384":
#         path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
#         state_dict = torch.load(path, map_location="cpu")
#         filtered_dict = checkpoint_filter_fn(state_dict, model)
#         model.load_state_dict(filtered_dict, strict=True)
#     elif "deit" in backbone:
#         load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
#     else:
#         load_custom_pretrained(model, default_cfg)
#
#     return model
#
#
# def create_decoder(encoder, decoder_cfg):
#     decoder_cfg = decoder_cfg.copy()
#     name = decoder_cfg.pop("name")
#     decoder_cfg["d_encoder"] = encoder.d_model
#     decoder_cfg["patch_size"] = encoder.patch_size
#
#     if "linear" in name:
#         decoder = DecoderLinear(**decoder_cfg)
#     elif name == "mask_transformer":
#         dim = encoder.d_model
#         n_heads = dim // 64
#         decoder_cfg["n_heads"] = n_heads
#         decoder_cfg["d_model"] = dim
#         decoder_cfg["d_ff"] = 4 * dim
#         decoder = MaskTransformer(**decoder_cfg)
#     else:
#         raise ValueError(f"Unknown decoder: {name}")
#     return decoder
#
#
# def create_segmenter(model_cfg):
#     model_cfg = model_cfg.copy()
#     decoder_cfg = model_cfg.pop("decoder")
#     decoder_cfg["n_cls"] = model_cfg["n_cls"]
#
#     encoder = create_vit(model_cfg)
#     decoder = create_decoder(encoder, decoder_cfg)
#     model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])
#
#     return model
