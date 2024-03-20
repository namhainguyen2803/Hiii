import os.path as osp
import ot
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from trainers.utils import SinkhornAlgorithm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.wasserstein_distance import *

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PLOT.N_CTX
        ctx_init = cfg.TRAINER.PLOT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.PLOT.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.PLOT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)  # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PLOT.CLASS_TOKEN_POSITION

    def forward(self):

        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device("cuda:0")
        self.device1 = torch.device("cuda")
        self.N = cfg.TRAINER.PLOT.N
        self.dataset = cfg.DATASET.NAME
        self.use_uniform = True
        self.eps = 0.1
        self.max_iter = 100

        self.text_feature_embed = nn.Sequential(nn.Linear(1024, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 256))

        self.visual_feature_embed = nn.Sequential(nn.Linear(1024, 256),
                                                  nn.ReLU(),
                                                  nn.Linear(256, 256))

    def formulate_OT_cosine_distance(self, image_features, text_features):
        M = image_features.shape[0]
        b = image_features.shape[1]

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()

        sim = sim.view(M, self.N, b * self.n_cls)
        sim = sim.permute(2, 0, 1)
        wdist = 1.0 - sim

        p = torch.zeros(b * self.n_cls, M, dtype=wdist.dtype, device=wdist.device).fill_(1. / M)
        q = torch.zeros(b * self.n_cls, self.N, dtype=wdist.dtype, device=wdist.device).fill_(1. / self.N)
        sinkhorn_solver = SinkhornAlgorithm(epsilon=self.eps, iterations=self.max_iter)
        with torch.no_grad():
            T = sinkhorn_solver(p, q, wdist)

        sim_op = torch.sum(T * wdist, dim=(1, 2))  # change here
        sim_op = sim_op.contiguous().view(b, self.n_cls)

        ot_distance = self.logit_scale.exp() * sim_op

        return ot_distance

    def formulate_OT_Wasserstein_distance(self, image_features, text_features):
        # image_features.shape == [49, 32, 1024]
        # text_features.shape == [4, 102, 1024]

        image_features = image_features.permute(1, 0, 2)  # image_features.shape == [32, 49, 1024]
        text_features = text_features.permute(1, 0, 2)  # text_features.shape == [102, 4, 1024]

        num_samples = image_features.shape[0]
        num_classes = text_features.shape[0]
        ot_distance = torch.zeros(num_samples, num_classes).to(self.device)
        for i in range(num_samples):
            for j in range(num_classes):
                # ot_distance[i, j] = sliced_wasserstein_distance(sources_samples=self.visual_feature_embed(image_features[i, :, :]),
                #                                                 target_samples=self.text_feature_embed(text_features[j, :, :]),
                #                                                 num_projections=500,
                #                                                 p=2,
                #                                                 device=self.device)

                ot_distance[i, j] = sliced_wasserstein_distance(sources_samples=image_features[i, :, :],
                                                                target_samples=text_features[j, :, :],
                                                                num_projections=500,
                                                                p=2,
                                                                device=self.device)

        return ot_distance

    def forward(self, image):
        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))  # [50, 32, 1024]
        image_features = image_features[1:]  # [49, 32, 1024]
        M = image_features.shape[0]
        self.d = image_features.shape[-1]
        # b: 32
        # M: 49
        # self.d: 1024
        # self.N: 4

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.contiguous().view(self.N, self.n_cls, self.d)

        # image_features = F.normalize(image_features, dim=2)
        # text_features = F.normalize(text_features, dim=2)
        # image_features.shape == [49, 32, 1024]
        # text_features.shape == [4, 102, 1024]
        # print(image_features.shape, text_features.shape)

        return self.formulate_OT_Wasserstein_distance(image_features=image_features.float(),
                                                      text_features=text_features.float())


@TRAINER_REGISTRY.register()
class PLOT(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PLOT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PLOT.PREC == "fp32" or cfg.TRAINER.PLOT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "text_feature_embed" not in name and "visual_feature_embed" not in name:
                print(f"Not require grad: {name}")
                param.requires_grad_(False)
            else:
                print(f"require grad: {name}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        if cfg.DATASET.NAME == "ImageNet":
            self.device = torch.device("cuda:0")
            # device0 = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim_prompt = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched_prompt = build_lr_scheduler(self.optim_prompt, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim_prompt, self.sched_prompt)

        self.optim_text = build_optimizer(self.model.text_feature_embed, cfg.OPTIM)
        self.sched_text = build_lr_scheduler(self.optim_text, cfg.OPTIM)
        self.register_model("text_feature_learner", self.model.text_feature_embed, self.optim_text, self.sched_text)

        self.optim_visual = build_optimizer(self.model.visual_feature_embed, cfg.OPTIM)
        self.sched_visual = build_lr_scheduler(self.optim_visual, cfg.OPTIM)
        self.register_model("visual_feature_learner", self.model.visual_feature_embed, self.optim_visual, self.sched_visual)

        self.scaler = GradScaler() if cfg.TRAINER.PLOT.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        # label.shape == [32]
        # image.shape == [32, 3, 224, 224]

        output = self.model(image)
        print(torch.sum(output))
        loss = F.cross_entropy(-output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(-output, label)[0].item(),
        }

        # ot_distance = self.model(image)  # shape == [32, 102]
        # batch_size = ot_distance.shape[0]
        # num_classes = ot_distance.shape[1]
        # reg = 0.01
        # a = torch.ones(batch_size).to(self.device)
        # b = torch.ones(num_classes).to(self.device)
        # T_empirical = torch.zeros(batch_size, num_classes).to(self.device)
        # for i in range(len(label)):
        #     cls = int(label[i].item())
        #     T_empirical[i, cls] += 1
        #     # b[cls] += 1
        # a = a / a.sum()
        # b = b / b.sum()
        # T_empirical = T_empirical / T_empirical.sum()
        # ot_distance = ot_distance / ot_distance.max()
        # reg_kl = (float("inf"), 0.01)
        # T_opt = ot.unbalanced.sinkhorn_unbalanced(a=a.float(), b=b.float(), reg=reg, reg_m=reg_kl,
        #                                           M=ot_distance.float(), numItermax=10000, method="sinkhorn_stabilized")
        # print(T_opt.sum())
        # # IOT
        # loss = -T_empirical * torch.log(T_opt + 1e-8)
        # loss = torch.sum(loss)
        # self.model_backward_and_update(loss)
        #
        # pred = torch.argmax(-ot_distance, dim=1)
        # print(f"Acc1: {torch.sum(pred == label) / len(pred)}")
        # loss_summary = {
        #     "loss": loss.item(),
        #     "acc": compute_accuracy(T_opt, label)[0].item(),
        # }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, image, label):
        ot_distance = self.model(image)  # shape == [32, 102]

        # batch_size = ot_distance.shape[0]
        # num_classes = ot_distance.shape[1]
        # reg = 0.01
        # a = torch.ones(batch_size).to(self.device)
        # b = torch.zeros(num_classes).to(self.device)
        # for i in range(len(label)):
        #     cls = int(label[i].item())
        #     b[cls] += 1
        # a = a / a.sum()
        # b = b / b.sum()
        # ot_distance = ot_distance / ot_distance.max()
        #
        # dist = ot.sinkhorn(a=a.float(), b=b.float(), M=ot_distance.float(), numItermax=10000, reg=reg, method="sinkhorn_stabilized")

        return -ot_distance

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
