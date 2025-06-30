import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder, uniEncoder, uniDecoder, LoRADecoder, LoRAEncoder,LoRADecoder2, LoRAEncoder2
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
import loralib as lora

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL_rimage(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # torch.Size([1, 8                                                                                                                                                                                                                          , 32, 32])
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()  # torch.Size([1, 4, 32, 32])
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class AutoencoderKL_rimage_finetune_(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        for param in self.parameters():  
            param.requires_grad = False
        for param in self.loss.parameters(): 
            param.requires_grad = True
        for param in self.decoder.conv_out.parameters(): 
            param.requires_grad = True
        for param in self.decoder.mid.attn_1.parameters():
            param.requires_grad = True
        for param in self.encoder.mid.attn_1.parameters():
            param.requires_grad = True
        # for param in self.encoder.mid
        # trainable_modelus = []
        # for name, para in self.named_parameters():  
        #     parts = name.split('.')  
        #     if len(parts) > 1: 
        #         modified_name = '.'.join(parts[:-1])  
        #     else:  
        #         modified_name = name  
        #     if ("norm" not in modified_name) and ("loss" not in modified_name):  
        #         trainable_modelus.append(modified_name)  
        # self.trainable_modelus =  trainable_modelus
        # peft_config = LoraConfig(
        #     task_type=TaskType.FEATURE_EXTRACTION, 
        #     r=8, lora_alpha=8, lora_dropout=0.1,
        #     target_modules=self.trainable_modelus,
        #     bias="none"
        # )
        
        # self = get_peft_model(self, peft_config)
        # self.print_trainable_parameters()
        # for name, param in self.named_parameters():  
        #     param.requires_grad = False
        # for name, param in self.named_parameters():  
        #     parts = name.split('.')  
        #     if len(parts) > 1: 
        #         modified_name = '.'.join(parts[:-1])  
        #     else:  
        #         modified_name = name  
        #     if ("norm" not in modified_name) and ("loss" not in modified_name): 
        #     # if any(modified_name in name for modified_name in self.trainable_modelus):  
        #         param.requires_grad = True
            
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # torch.Size([1, 8                                                                                                                                                                                                                          , 32, 32])
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()  # torch.Size([1, 4, 32, 32])
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", loss_signs=loss_signs)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",loss_signs=loss_signs)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKL_rimage_finetune(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        for param in self.parameters():  
            param.requires_grad = False
        for param in self.loss.parameters(): 
            param.requires_grad = True
        for param in self.decoder.conv_out.parameters(): 
            param.requires_grad = True
        for param in self.decoder.mid.attn_1.parameters():
            param.requires_grad = True
        # for param in self.encoder.mid.attn_1.parameters():
        #     param.requires_grad = True
        # for param in self.encoder.mid
        # trainable_modelus = []
        # for name, para in self.named_parameters():  
        #     parts = name.split('.')  
        #     if len(parts) > 1: 
        #         modified_name = '.'.join(parts[:-1])  
        #     else:  
        #         modified_name = name  
        #     if ("norm" not in modified_name) and ("loss" not in modified_name):  
        #         trainable_modelus.append(modified_name)  
        # self.trainable_modelus =  trainable_modelus
        # peft_config = LoraConfig(
        #     task_type=TaskType.FEATURE_EXTRACTION, 
        #     r=8, lora_alpha=8, lora_dropout=0.1,
        #     target_modules=self.trainable_modelus,
        #     bias="none"
        # )
        
        # self = get_peft_model(self, peft_config)
        # self.print_trainable_parameters()
        # for name, param in self.named_parameters():  
        #     param.requires_grad = False
        # for name, param in self.named_parameters():  
        #     parts = name.split('.')  
        #     if len(parts) > 1: 
        #         modified_name = '.'.join(parts[:-1])  
        #     else:  
        #         modified_name = name  
        #     if ("norm" not in modified_name) and ("loss" not in modified_name): 
        #     # if any(modified_name in name for modified_name in self.trainable_modelus):  
        #         param.requires_grad = True
            
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # torch.Size([1, 8                                                                                                                                                                                                                          , 32, 32])
        if sample_posterior:
            z = posterior.sample()
            z = posterior.mode()
        else:
            z = posterior.mode()  # torch.Size([1, 4, 32, 32])
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", loss_signs=loss_signs)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",loss_signs=loss_signs)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class AutoencoderKL_rimage_finetuneD(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        for param in self.parameters():  
            param.requires_grad = False
        for param in self.loss.parameters(): 
            param.requires_grad = True
        for param in self.decoder.conv_out.parameters(): 
            param.requires_grad = True
        for param in self.decoder.mid.attn_1.parameters():
            param.requires_grad = True
        for param in self.decoder.mid.block_2.parameters():
            param.requires_grad = True
        for param in self.decoder.mid.block_1.parameters():
            param.requires_grad = True
            
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # torch.Size([1, 8                                                                                                                                                                                                                          , 32, 32])
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()  # torch.Size([1, 4, 32, 32])
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        inputs_m =  self.get_input(batch, 'image_m') #调整后的
        reconstructions, posterior = self(inputs_m)
        
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", loss_signs=loss_signs)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",loss_signs=loss_signs)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        inputs_m =  self.get_input(batch, 'image_m') #调整后的
        reconstructions, posterior = self(inputs_m)
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)

        inputs_m =  self.get_input(batch, 'image_m') #调整后的
        inputs_m = inputs_m.to(self.device)

        if not only_inputs:
            xrec, posterior = self(inputs_m)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
                inputs_m = self.to_rgb(inputs_m)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        log["inputs_m"] = inputs_m


        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class tm(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.m1_encoder = self.encoder
        self.m1_decoder = self.decoder
        self.m2_encoder = self.encoder
        self.m2_decoder = self.decoder
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.m1_quant_conv = self.quant_conv
        self.m1_post_quant_conv = self.post_quant_conv
        self.m2_quant_conv = self.quant_conv
        self.m2_post_quant_conv = self.post_quant_conv
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        for param in self.parameters():  
            param.requires_grad = False
        for param in self.loss.parameters(): 
            param.requires_grad = True
        for param in self.m1_decoder.conv_out.parameters(): 
            param.requires_grad = True
        for param in self.m2_decoder.conv_out.parameters(): 
            param.requires_grad = True
        for param in self.m1_decoder.mid.attn_1.parameters():
            param.requires_grad = True
        for param in self.m2_decoder.mid.attn_1.parameters():
            param.requires_grad = True
        for param in self.m1_encoder.mid.attn_1.parameters():
            param.requires_grad = True
        for param in self.m2_encoder.mid.attn_1.parameters():
            param.requires_grad = True
            
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def m1_encode(self, x):
        h = self.m1_encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.m1_quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def m1_decode(self, z):
        z = self.m1_post_quant_conv(z)
        dec = self.m1_decoder(z)
        return dec

    def m2_encode(self, x):
        h = self.m2_encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.m2_quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def m2_decode(self, z):
        z = self.m2_post_quant_conv(z)
        dec = self.m2_decoder(z)
        return dec
    
    def forward(self, m1_input, m2_input, sample_posterior=True):
        decs, posteriors = [], []
        m1_posterior = self.m1_encode(m1_input) 
        posteriors.append(m1_posterior)
        m2_posterior = self.m2_encode(m2_input)
        posteriors.append(m2_posterior)

        if sample_posterior:
            m1_z = m1_posterior.sample()
            m2_z = m2_posterior.sample()
        else:
            m1_z = m1_posterior.mode() 
            m2_z = m2_posterior.mode()

        m1_dec = self.m1_decode(m1_z)
        decs.append(m1_dec)
        m2_dec = self.m2_decode(m2_z)
        decs.append(m2_dec)
        return decs, posteriors

    def get_input(self, batch, k):
        image, mask = batch[k], batch['c_mask']
        if len(image.shape) == 3:
            image = image[..., None]
        image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if len(mask.shape) == 3:
            mask = mask[..., None]
        mask = mask.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return image, mask

    def training_step(self, batch, batch_idx, optimizer_idx):
        m1_input, m2_input = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(m1_input, m2_input)
        m1_reconstruction, m2_reconstruction = reconstructions[0], reconstructions[1]
        m1_posterior, m2_posterior = posterior[0], posterior[1]
        
        # loss_signs= []
        # for i in range(len(batch['file_path_'])):
        #     if batch['file_path_'][i].split('/')[6] == 'mask':
        #        loss_signs.append('mask')
        #     else:
        #        loss_signs.append('image')
        
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            m1_aeloss, m1_log_dict_ae = self.loss(m1_input, m1_reconstruction, m1_posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_m1_last_layer(), split="train")
            m2_aeloss, m2_log_dict_ae = self.loss(m2_input, m2_reconstruction, m2_posterior, optimizer_idx, self.global_step, 
                                                last_layer=self.get_m2_last_layer(), split="train")
            
            aeloss = m1_aeloss + m2_aeloss
            log_dict_ae = {}
            for key in m1_log_dict_ae.keys():  
                value = m1_log_dict_ae[key] + m2_log_dict_ae[key]  
                log_dict_ae[key] = value 
            
            KL_m1m2 = m1_posterior.kl(m2_posterior) 
            KL_m2m1 = m2_posterior.kl(m1_posterior)
            KL_loss = torch.sum(KL_m1m2 + KL_m2m1,dim=[0])/KL_m1m2.shape[0]
            w_KL_loss_inModalities = 0
            self.log("KL_loss_inModalities", w_KL_loss_inModalities* KL_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss + w_KL_loss_inModalities * KL_loss
        
        if optimizer_idx == 1:
            # train the discriminator
            m1_discloss, m1_log_dict_disc = self.loss(m1_input, m1_reconstruction, m1_posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_m1_last_layer(), split="train") 
            m2_discloss, m2_log_dict_disc = self.loss(m2_input, m2_reconstruction, m2_posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_m2_last_layer(), split="train")
            
            discloss = m1_discloss + m2_discloss
            log_dict_disc = {}
            for key in m1_log_dict_disc.keys():  
                value = m1_log_dict_disc[key] + m2_log_dict_disc[key]  
                log_dict_disc[key] = value 
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        m1_input, m2_input = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(m1_input, m2_input)
        m1_reconstruction, m2_reconstruction = reconstructions[0], reconstructions[1]
        m1_posterior, m2_posterior = posterior[0], posterior[1]

        m1_aeloss, m1_log_dict_ae = self.loss(m1_input, m1_reconstruction, m1_posterior, 0, self.global_step,
                                            last_layer=self.get_m1_last_layer(), split="val")
        m2_aeloss, m2_log_dict_ae = self.loss(m2_input, m2_reconstruction, m2_posterior, 0, self.global_step, 
                                            last_layer=self.get_m2_last_layer(), split="val")

        m1_discloss, m1_log_dict_disc = self.loss(m1_input, m1_reconstruction, m1_posterior, 1, self.global_step,
                                            last_layer=self.get_m1_last_layer(), split="val") 
        m2_discloss, m2_log_dict_disc = self.loss(m2_input, m2_reconstruction, m2_posterior, 1, self.global_step,
                                            last_layer=self.get_m2_last_layer(), split="val")
        
        aeloss = m1_aeloss + m2_aeloss
        discloss = m1_discloss + m2_discloss
        KL_m1m2 = m1_posterior.kl(m2_posterior) 
        KL_m2m1 = m2_posterior.kl(m1_posterior)
        KL_loss = torch.sum(KL_m1m2 + KL_m2m1,dim=[0])/KL_m1m2.shape[0]

        w_KL_loss_inModalities = 0.1
        self.log("KL_loss_inModalities", w_KL_loss_inModalities* KL_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        log_dict_ae, log_dict_disc = {}, {} 
        
        for key in m1_log_dict_ae.keys():  
            value = m1_log_dict_ae[key] + m2_log_dict_ae[key]  
            log_dict_ae[key] = value 

        for key in m1_log_dict_disc.keys():  
            value = m1_log_dict_disc[key] + m2_log_dict_disc[key]  
            log_dict_disc[key] = value 

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def get_m1_last_layer(self):
        return self.m1_decoder.conv_out.weight

    def get_m2_last_layer(self):
        return self.m2_decoder.conv_out.weight
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        m1_input, m2_input = self.get_input(batch, self.image_key)
        m1_input, m2_input = m1_input, m2_input.to(self.device)
        if not only_inputs:
            # xrec, posterior = self(x)
            reconstructions, posterior = self(m1_input, m2_input)
            m1_reconstruction, m2_reconstruction = reconstructions[0], reconstructions[1]
            m1_posterior, m2_posterior = posterior[0], posterior[1]
            if m1_input.shape[1] > 3:
                # colorize with random projection
                assert m1_reconstruction.shape[1] > 3
                m1_input = self.to_rgb(m1_input)
                m1_reconstruction = self.to_rgb(m1_reconstruction)
            log["polyp_samples"] = self.m1_decode(torch.randn_like(m1_posterior.sample()))
            log["polyp_reconstructions"] = m1_reconstruction
            if m2_input.shape[1] > 3:
                # colorize with random projection
                assert m2_reconstruction.shape[1] > 3
                m2_input = self.to_rgb(m2_input)
                m2_reconstruction = self.to_rgb(m2_reconstruction)
            log["mask_samples"] = self.m2_decode(torch.randn_like(m2_posterior.sample()))
            log["mask_reconstructions"] = m2_reconstruction
        log["polyp_inputs"] = m1_input
        log["mask_inputs"] = m2_input
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKL_bifinetune_polyp(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.m1_encoder = self.encoder
        self.m1_decoder = self.decoder
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.m1_quant_conv = self.quant_conv
        self.m1_post_quant_conv = self.post_quant_conv

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        for param in self.parameters():  
            param.requires_grad = False
        for param in self.loss.parameters(): 
            param.requires_grad = True
        for param in self.m1_decoder.conv_out.parameters(): 
            param.requires_grad = True
        for param in self.m1_decoder.mid.attn_1.parameters():
            param.requires_grad = True
        for param in self.m1_encoder.mid.attn_1.parameters():
            param.requires_grad = True

        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.m1_encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.m1_quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def decode(self, z):
        z = self.m1_post_quant_conv(z)
        dec = self.m1_decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # torch.Size([1, 8                                                                                                                                                                                                                          , 32, 32])
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()  # torch.Size([1, 4, 32, 32])
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", loss_signs=loss_signs)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",loss_signs=loss_signs)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKL_bifinetune_mask(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.m2_encoder = self.encoder
        self.m2_decoder = self.decoder
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.m2_quant_conv = self.quant_conv
        self.m2_post_quant_conv = self.post_quant_conv

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        for param in self.parameters():  
            param.requires_grad = False
        for param in self.loss.parameters(): 
            param.requires_grad = True
        for param in self.m2_decoder.conv_out.parameters(): 
            param.requires_grad = True
        for param in self.m2_decoder.mid.attn_1.parameters():
            param.requires_grad = True
        for param in self.m2_encoder.mid.attn_1.parameters():
            param.requires_grad = True

        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.m2_encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.m2_quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def decode(self, z):
        z = self.m2_post_quant_conv(z)
        dec = self.m2_decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # torch.Size([1, 8                                                                                                                                                                                                                          , 32, 32])
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()  # torch.Size([1, 4, 32, 32])
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", loss_signs=loss_signs)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",loss_signs=loss_signs)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        loss_signs= []
        for i in range(len(batch['file_path_'])):
            if batch['file_path_'][i].split('/')[6] == 'mask':
               loss_signs.append('mask')
            else:
               loss_signs.append('image')
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val",loss_signs=loss_signs)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKL_rimage_LoRA(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = LoRAEncoder(**ddconfig)
        self.decoder = LoRADecoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        lora.mark_only_lora_as_trainable(self)
        for param in self.loss.parameters():  
            param.requires_grad = True
        for param in self.decoder.conv_out.parameters():
            param.requires_grad = True

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # torch.Size([1, 8                                                                                                                                                                                                                          , 32, 32])
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()  # torch.Size([1, 4, 32, 32])
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    def get_last_layer(self):
        # self.decoder.conv_out.weight.requires_grad = True
        return self.decoder.conv_out.weight

        # return self.decoder.lora_out.conv.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKL_rmask(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = uniEncoder(**ddconfig)
        self.decoder = uniDecoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # torch.Size([1, 8                                                                                                                                                                                                                          , 32, 32])
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()  # torch.Size([1, 4, 32, 32])
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch['c_mask']
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


class mmAutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.m1_encoder = Encoder(**ddconfig) # modality1
        self.m1_decoder = Decoder(**ddconfig)
        self.m2_encoder = uniEncoder(**ddconfig) # modality2
        self.m2_decoder = uniDecoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.m1_quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.m1_post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.m2_quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.m2_post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def m1_encode(self, x):
        h = self.m1_encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.m1_quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def m1_decode(self, z):
        z = self.m1_post_quant_conv(z)
        dec = self.m1_decoder(z)
        return dec

    def m2_encode(self, x):
        h = self.m2_encoder(x) # output, torch.Size([1, 8, 32, 32])
        moments = self.m2_quant_conv(h) # output, torch.Size([1, 8, 32, 32]
        posterior = DiagonalGaussianDistribution(moments) # output, torch.Size([1, 4, 32, 32])
        return posterior # class DiagonalGaussianDistribution()

    def m2_decode(self, z):
        z = self.m2_post_quant_conv(z)
        dec = self.m2_decoder(z)
        return dec

    def forward(self, m1_input, m2_input, sample_posterior=True):
        decs, posteriors = [], []
        m1_posterior = self.m1_encode(m1_input)  # torch.Size([1, 8                                                                                                                                                                                                                          , 32, 32])
        posteriors.append(m1_posterior)
        if m2_input.shape[1]==3:
            print('')
        elif m2_input.shape[1]==1:
            print('')
        else:
            print('')
        m2_posterior = self.m2_encode(m2_input)
        posteriors.append(m2_posterior)

        if sample_posterior:
            m1_z = m1_posterior.sample()
            m2_z = m2_posterior.sample()
        else:
            m1_z = m1_posterior.mode()  # torch.Size([1, 4, 32, 32])
            m2_z = m2_posterior.mode()
        
        # z = m1_z + m2_z
        m1_dec = self.m1_decode(m1_z)
        decs.append(m1_dec)
        m2_dec = self.m2_decode(m2_z)
        decs.append(m2_dec)
        return decs, posteriors

    def get_input(self, batch, k): # wait for modification.
        image, mask = batch[k], batch['c_mask']
        if len(image.shape) == 3:
            image = image[..., None]
        image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if len(mask.shape) == 3:
            mask = mask[..., None]
        mask = mask.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return image, mask

    def training_step(self, batch, batch_idx, optimizer_idx):
        m1_input, m2_input = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(m1_input, m2_input)
        m1_reconstruction, m2_reconstruction = reconstructions[0], reconstructions[1]
        m1_posterior, m2_posterior = posterior[0], posterior[1]

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            m1_aeloss, m1_log_dict_ae = self.loss(m1_input, m1_reconstruction, m1_posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_m1_last_layer(), split="train")
            m2_aeloss, m2_log_dict_ae = self.loss(m2_input, m2_reconstruction, m2_posterior, optimizer_idx, self.global_step, 
                                                last_layer=self.get_m2_last_layer(), split="train")
            
            aeloss = m1_aeloss + m2_aeloss
            log_dict_ae = {}
            for key in m1_log_dict_ae.keys():  
                value = m1_log_dict_ae[key] + m2_log_dict_ae[key]  
                log_dict_ae[key] = value 
            
            KL_m1m2 = m1_posterior.kl(m2_posterior) 
            KL_m2m1 = m2_posterior.kl(m1_posterior)
            KL_loss = torch.sum(KL_m1m2 + KL_m2m1,dim=[0])/KL_m1m2.shape[0]
            self.log("KL_loss_inModalities", 100* KL_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss + 10 * KL_loss

        if optimizer_idx == 1:
            # train the discriminator
            m1_discloss, m1_log_dict_disc = self.loss(m1_input, m1_reconstruction, m1_posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_m1_last_layer(), split="train") 
            m2_discloss, m2_log_dict_disc = self.loss(m2_input, m2_reconstruction, m2_posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_m2_last_layer(), split="train")
            
            discloss = m1_discloss + m2_discloss
            log_dict_disc = {}
            for key in m1_log_dict_disc.keys():  
                value = m1_log_dict_disc[key] + m2_log_dict_disc[key]  
                log_dict_disc[key] = value 
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        m1_input, m2_input = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(m1_input, m2_input)
        m1_reconstruction, m2_reconstruction = reconstructions[0], reconstructions[1]
        m1_posterior, m2_posterior = posterior[0], posterior[1]

        m1_aeloss, m1_log_dict_ae = self.loss(m1_input, m1_reconstruction, m1_posterior, 0, self.global_step,
                                            last_layer=self.get_m1_last_layer(), split="val")
        m2_aeloss, m2_log_dict_ae = self.loss(m2_input, m2_reconstruction, m2_posterior, 0, self.global_step, 
                                            last_layer=self.get_m2_last_layer(), split="val")

        m1_discloss, m1_log_dict_disc = self.loss(m1_input, m1_reconstruction, m1_posterior, 1, self.global_step,
                                            last_layer=self.get_m1_last_layer(), split="val") 
        m2_discloss, m2_log_dict_disc = self.loss(m2_input, m2_reconstruction, m2_posterior, 1, self.global_step,
                                            last_layer=self.get_m2_last_layer(), split="val")
        
        aeloss = m1_aeloss + m2_aeloss
        discloss = m1_discloss + m2_discloss
        KL_m1m2 = m1_posterior.kl(m2_posterior) 
        KL_m2m1 = m2_posterior.kl(m1_posterior)
        KL_loss = torch.sum(KL_m1m2 + KL_m2m1,dim=[0])/KL_m1m2.shape[0]
        self.log("KL_loss", KL_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        log_dict_ae, log_dict_disc = {}, {} 
        
        for key in m1_log_dict_ae.keys():  
            value = m1_log_dict_ae[key] + m2_log_dict_ae[key]  
            log_dict_ae[key] = value 

        for key in m1_log_dict_disc.keys():  
            value = m1_log_dict_disc[key] + m2_log_dict_disc[key]  
            log_dict_disc[key] = value 

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.m1_encoder.parameters())+
                                  list(self.m1_decoder.parameters())+
                                  list(self.m1_quant_conv.parameters())+
                                  list(self.m1_post_quant_conv.parameters())+
                                  list(self.m2_encoder.parameters())+
                                  list(self.m2_decoder.parameters())+
                                  list(self.m2_quant_conv.parameters())+
                                  list(self.m2_post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_m1_last_layer(self):
        return self.m1_decoder.conv_out.weight

    def get_m2_last_layer(self):
        return self.m2_decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        m1_input, m2_input = self.get_input(batch, self.image_key)
        m1_input, m2_input = m1_input, m2_input.to(self.device)
        if not only_inputs:
            # xrec, posterior = self(x)
            reconstructions, posterior = self(m1_input, m2_input)
            m1_reconstruction, m2_reconstruction = reconstructions[0], reconstructions[1]
            m1_posterior, m2_posterior = posterior[0], posterior[1]
            if m1_input.shape[1] > 3:
                # colorize with random projection
                assert m1_reconstruction.shape[1] > 3
                m1_input = self.to_rgb(m1_input)
                m1_reconstruction = self.to_rgb(m1_reconstruction)
            log["m1_samples"] = self.m1_decode(torch.randn_like(m1_posterior.sample()))
            log["m1_reconstructions"] = m1_reconstruction
            if m2_input.shape[1] > 3:
                # colorize with random projection
                assert m2_reconstruction.shape[1] > 3
                m2_input = self.to_rgb(m2_input)
                m2_reconstruction = self.to_rgb(m2_reconstruction)
            log["m2_samples"] = self.m2_decode(torch.randn_like(m2_posterior.sample()))
            log["m2_reconstructions"] = m2_reconstruction
        log["m1_inputs"] = m1_input
        log["m2_inputs"] = m2_input
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


# Base MMVAEplus class definition
import torch
import torch.nn as nn

def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean

class MMVAEplus(nn.Module):
    """
    MMVAEplus class definition.
    """
    def __init__(self, prior_dist, params, *vaes):
        super(MMVAEplus, self).__init__()
        self.pz = prior_dist # Prior distribution
        self.pw = prior_dist
        self.vaes = nn.ModuleList([vae(params) for vae in vaes]) # List of unimodal VAEs (one for each modality)
        self.modelName = None  # Filled-in in subclass
        self.params = params # Model parameters (i.e. args passed to main script)

    @staticmethod
    def getDataSets(batch_size, shuffle=True, device="cuda"):
        # Handle getting individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        """
        Forward function.
        Input:
            - x: list of data samples for each modality
            - K: number of samples for reparameterization in latent space

        Returns:
            - qu_xs: List of encoding distributions (one per encoder)
            - px_us: Matrix of self- and cross- reconstructions. px_zs[m][n] contains
                    m --> n  reconstruction.
            - uss: List of latent codes, one for each modality. uss[m] contains latents inferred
                   from modality m. Note there latents are the concatenation of private and shared latents.
        """
        qu_xs, uss = [], []
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        # Loop over unimodal vaes
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(x[m], K=K) # Get Encoding dist, Decoding dist, Latents for unimodal VAE m modality
            qu_xs.append(qu_x) # Append encoding distribution to list
            uss.append(us) # Append latents to list
            px_us[m][m] = px_u  # Fill-in self-reconstructions in the matrix
        # Loop over unimodal vaes and compute cross-modal reconstructions
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal with cross-modal reconstructions
                    # Get shared latents from encoding modality e
                    _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                    # Resample modality-specific encoding from modality-specific auxiliary distribution for decoding modality m
                    pw = vae.pw(*vae.pw_params_aux)
                    latents_w = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                    # Fixed for cuda (sorry)
                    if not self.params.no_cuda and torch.cuda.is_available():
                        latents_w.cuda()
                    # Combine shared and resampled private latents
                    us_combined = torch.cat((latents_w, z_e), dim=-1)
                    # Get cross-reconstruction likelihood
                    px_us[e][d] = vae.px_u(*vae.dec(us_combined))
        return qu_xs, px_us, uss


    def generate_unconditional(self, N):
        """
        Unconditional generation.
        Args:
            N: Number of samples to generate.
        Returns:
            Generations
        """
        with torch.no_grad():
            data = []
            # Sample N shared latents
            pz = self.pz(*self.pz_params)
            latents_z = pz.rsample(torch.Size([N]))
            # Decode for all modalities
            for d, vae in enumerate(self.vaes):
                pw = self.pw(*self.pw_params)
                latents_w = pw.rsample([latents_z.size()[0]])
                latents = torch.cat((latents_w, latents_z), dim=-1)
                px_u = vae.px_u(*vae.dec(latents))
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality


    # def reconstruct(self, data):
    #     """
    #     Test-time reconstruction
    #     Args:
    #         data: Input
    #
    #     Returns:
    #         Reconstructions
    #     """
    #     with torch.no_grad():
    #         _, px_zs, _ = self.forward(data)
    #         # cross-modal matrix of reconstructions
    #         recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
    #     return recons

    def self_and_cross_modal_generation_forward(self, data, K=1):
        """
        Test-time self- and cross-model generation forward function.
        Args:
            data: Input

        Returns:
            Unimodal encoding distribution, Matrix of self- and cross-modal reconstruction distrubutions, Latent embeddings

        """
        qu_xs, uss = [], []
        # initialise cross-modal matrix
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(data[m], K=K)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # fill-in diagonal
        for e, us in enumerate(uss):
            latents_w, latents_z = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
            for d, vae in enumerate(self.vaes):
                mean_w, scale_w = self.pw_params
                # Tune modality-specific std prior
                # scale_w = factor * scale_w
                pw = self.pw(mean_w, scale_w)
                latents_w_new = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                us_new = torch.cat((latents_w_new, latents_z), dim=-1)
                if e != d:  # fill-in off-diagonal
                    px_us[e][d] = vae.px_u(*vae.dec(us_new))
        return qu_xs, px_us, uss

    def self_and_cross_modal_generation(self, data):
        """
        Test-time self- and cross-reconstruction.
        Args:
            data: Input

        Returns:
            Matrix of self- and cross-modal reconstructions

        """
        with torch.no_grad():
            _, px_us, _ = self.self_and_cross_modal_generation_forward(data)
            # ------------------------------------------------
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_u) for px_u in r] for r in px_us]
        return recons

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

