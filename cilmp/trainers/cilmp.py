import copy
import os.path as osp
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES
from transformers.activations import ACT2FN

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT,
                          "classnames": cfg.classnames,
                          'prefix_length': cfg.TRAINER.PROMPTSRC.PREFIX_LENGTH,
                          'suffix_length': cfg.TRAINER.PROMPTSRC.SUFFIX_LENGTH,
                          'low_rank_dimension': cfg.TRAINER.PROMPTSRC.LOW_RANK_DIMENSION
                          }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model



class NoreftIntervention(nn.Module):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, embed_dim, low_rank_dimension):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, low_rank_dimension, bias=True)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, low_rank_dimension)
        self.dropout = torch.nn.Dropout(0.1)
        self.act_fn = ACT2FN["linear"]
        
    def forward(self, x):
        x_dtype = x.dtype
        x = x.to(torch.float32)
        proj_base = self.proj_layer(x)
        output = x + torch.matmul(
            (self.act_fn(self.learned_source(x)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(x_dtype))



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, img_features=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, img_features)
        if type(x) == dict:
            x = x['x']

        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        if img_features is not None:
            bs = img_features.shape[0]
            n_cls, len_tokenized_prompts = tokenized_prompts.shape
            tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(bs, 1, 1)
            tokenized_prompts = tokenized_prompts.view(-1, len_tokenized_prompts)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class LowRankRotateLayer(nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LoreftIntervention(nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, embed_dim, low_rank_dimension):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = rotate_layer
        self.learned_source = torch.nn.Linear(embed_dim, low_rank_dimension)
        self.dropout = torch.nn.Dropout(0.1)
        self.act_fn = ACT2FN["linear"]
        
    def forward(self, base):
        base_dtype = base.dtype
        base = base.to(torch.float32)
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base_dtype))


class ConditionalLoreftIntervention(nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, embed_dim, low_rank_dimension, ctx_dim, n_cls):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = rotate_layer
        self.learned_source = torch.nn.Linear(embed_dim * 2, low_rank_dimension)
        self.dropout = torch.nn.Dropout(0.1)
        self.act_fn = ACT2FN["linear"]
        self.image_feature_learned_weight = nn.Sequential(
            torch.nn.Linear(ctx_dim, low_rank_dimension),
            torch.nn.Linear(low_rank_dimension, embed_dim)
        )
        self.n_cls = n_cls
        self.low_rank_dimension = low_rank_dimension

    def forward(self, base, img_feature):

        base_org = base.to(torch.float32)
        img_feature_shape = img_feature.shape  

        processed_img_feature = self.image_feature_learned_weight(img_feature.to(torch.float32)) 
        rd = processed_img_feature.unsqueeze(1) * base.to(torch.float32).unsqueeze(0)
        bs = rd.shape[0]
        base2 = torch.cat([base.unsqueeze(0).repeat(bs, 1, 1), rd], dim=-1)
        processed_base = self.learned_source(base2)
        base_shifted = processed_base
        base_dtype = base.dtype
        base = base.to(torch.float32)
        rotated_base = self.rotate_layer(base_org)
        output = []
        for base_shifted_i in base_shifted:
            tmp = base_org + torch.matmul(
                (self.act_fn(base_shifted_i) - rotated_base), self.rotate_layer.weight.T
            )
            output.append(self.dropout(tmp.to(base_dtype)))
        
        return torch.stack(output)



class LoraProjection(nn.Module):
    def __init__(self, in_dim, low_rank_dim, out_dim):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(low_rank_dim, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, low_rank_dim))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.to(torch.float32)
        after_A = F.linear(x, self.lora_A)
        after_B = F.linear(after_A, self.lora_B)
        return after_B.to(x_dtype)

class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        assert cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTSRC.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTSRC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)
        llm_hidden_representations = []
        llm_rep_length = []
        prefix_length = cfg.TRAINER.PROMPTSRC.PREFIX_LENGTH
        suffix_length = cfg.TRAINER.PROMPTSRC.SUFFIX_LENGTH
        llm_embed_dim = 4096
        low_rank_dimension = cfg.TRAINER.PROMPTSRC.LOW_RANK_DIMENSION
        max_len = 0
        min_len = 1000

        for name in classnames:
            if len(classnames) == 8:
                if "esophagitis" in classnames: # kvasir
                    path = osp.join('llm_representations/kvasir', name + '.pth')
                elif "lymphocyte" in classnames: # bloodmnist
                    path = osp.join('llm_representations/bloodmnist', name + '.pth')
                else: # odir
                    path = osp.join('llm_representations/odir', name + '.pth')
                

            elif len(classnames) == 7:
                # for dermamnist:
                if "actinic keratoses and intraepithelial carcinoma" in classnames:
                    path = osp.join('llm_representations/dermamnist', name + '.pth')
                else:
                    # for isic
                    path = osp.join('llm_representations/isic', name + '.pth')

            elif len(classnames) == 2:
                if "pneumonia" in classnames:
                    path = osp.join('llm_representations/pneumonia2', name + '.pth')
                elif "cataract" in classnames:
                    path = osp.join('llm_representations/odir2', name + '.pth')
                else:  # adam
                    path = osp.join('llm_representations/adam', name + '.pth')
            elif len(classnames) == 3:
                if "malignant" in classnames: # busi
                    path = osp.join('llm_representations/busi', name + '.pth')
                elif "covid" in classnames: # cpn_x_ray
                    path = osp.join('llm_representations/cpn_x_ray', name + '.pth')
                else:  # pneumonia3
                    path = osp.join('llm_representations/pneumonia3', name + '.pth')
            elif len(classnames) == 6: # fetal_us
                path = osp.join('llm_representations/fetal_us', name + '.pth')
            elif len(classnames) == 5: # aptos 2019
                if "nevus" in classnames:
                    # derm7pt
                    name = name.split(" ")[-1]
                    path = osp.join('llm_representations/derm7pt', name + '.pth')
                else:
                    # aptos 2019
                    path = osp.join('llm_representations/aptos2019', name + '.pth')
            elif len(classnames) == 4: # chaoyang
                path = osp.join('llm_representations/chaoyang', name + '.pth')
            else:
                name = name.split(" ")[-1]
                path = osp.join('llm_representations/derm7pt', name + '.pth')
            llm_hidden_rep = torch.load(path)
            max_len = max(max_len, len(llm_hidden_rep))
            min_len = min(min_len, len(llm_hidden_rep))

            llm_hidden_representations.append(llm_hidden_rep)
            llm_rep_length.append(len(llm_hidden_rep))
        
        self.llm_rep_length = llm_rep_length
        self.total_length = max_len
        self.min_rep_length = min_len

        classnames = [name.replace("_", " ") for name in classnames]
        add_llm_prompt_prefix = prompt_prefix
        add_llm_prompt_prefix = [add_llm_prompt_prefix + " " + " ".join(["H"] * cur_len) for cur_len in llm_rep_length]
        add_llm_prompts = [add_llm_prompt_prefix[i] + " " + name + "." for i, name in enumerate(classnames)]
        add_llm_tokenized_prompts = torch.cat([clip.tokenize(p) for p in add_llm_prompts])
        self.add_llm_tokenized_prompts = add_llm_tokenized_prompts
        self.llm_prompt = torch.empty(len(llm_hidden_representations), max_len, llm_embed_dim).to(ctx_vectors.dtype)
        for index, hr in enumerate(llm_hidden_representations):
            self.llm_prompt[index, :len(hr), :] = hr.to(ctx_vectors.dtype)
        self.llm_prompt = self.llm_prompt.cuda()
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.low_rank_dimension = low_rank_dimension
        n_cls = len(classnames)
        self.prefix_intervention = ConditionalLoreftIntervention(llm_embed_dim, low_rank_dimension, ctx_dim, n_cls)
        self.suffix_intervention = ConditionalLoreftIntervention(llm_embed_dim, low_rank_dimension, ctx_dim, n_cls)
        self.lora_proj = LoraProjection(llm_embed_dim, low_rank_dimension, ctx_dim)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            all_teacher_features = []
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        self.register_buffer("token_prefix", embedding[:, :1, :])  
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, llm_prompt, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompt = []
        for i in range(len(self.llm_rep_length)):
            prompt_i = torch.cat([
                prefix[i],
                ctx[i],
                llm_prompt[i, :self.llm_rep_length[i], :],
                suffix[i]
                ], dim=0).unsqueeze(0)[:, :77] 
            prompt.append(prompt_i)
        prompts = torch.cat(prompt, dim=0)
                
        return prompts

    def forward_llm(self, img_fea):
        prefix = self.llm_prompt[:, :self.prefix_length] 
        after_prefix_int = []
        for i in range(self.prefix_length):
            rep = prefix[:, i]  
            after_prefix_int.append(self.prefix_intervention(rep, img_fea).unsqueeze(-2)) 
        after_prefix_int = torch.cat(after_prefix_int, dim=-2)  

        bs = after_prefix_int.shape[0]

        if self.llm_rep_length[0] ==  self.llm_rep_length[-1]:
            total_length = self.llm_rep_length[0]
            suffix = self.llm_prompt[:, total_length - self.suffix_length:] 
            after_suffix_int = []

            for i in range(self.suffix_length):
                rep = suffix[:, i]  
                after_suffix_int.append(self.suffix_intervention(rep, img_fea).unsqueeze(-2)) 
            after_suffix_int = torch.cat(after_suffix_int, dim=-2)  
            unchanged = self.llm_prompt[:, self.prefix_length: total_length - self.suffix_length]  
            unchanged = unchanged.unsqueeze(0).repeat(bs, 1, 1, 1)  
            after_int = torch.cat([
                after_prefix_int,
                unchanged,
                after_suffix_int
            ], dim=-2)
            bs2, n_cls2, len_total2, dim2 = after_int.shape
            assert bs == bs2
            after_int = after_int.view(-1, len_total2, dim2)
            n_cls, max_len, dim = after_int.shape
            after_int = after_int.view(-1, dim)
            final_output = self.lora_proj(after_int)
            final_output = final_output.reshape(n_cls, max_len, -1)
            final_output_dim = final_output.shape[-1]
            final_output = final_output.view(bs2, n_cls2, len_total2, final_output_dim)
            return final_output

        else:
            after_suffix_int = []
            unchanged = []
            for i in range(len(self.llm_prompt)): 
                after_suffix_int_i = []
                suffix = self.llm_prompt[i, self.llm_rep_length[i] - self.suffix_length:, :].unsqueeze(0) 
                unchanged_i = self.llm_prompt[i, self.prefix_length:self.llm_rep_length[i] - self.suffix_length, :].unsqueeze(0)
                unchanged_i = unchanged_i.unsqueeze(0).repeat(bs, 1, 1, 1) 
                unchanged.append(unchanged_i)
                for j in range(self.suffix_length):
                    rep = suffix[:, j]  
                    after_suffix_int_i.append(self.suffix_intervention(rep, img_fea).unsqueeze(-2)) 
                after_suffix_int_i = torch.cat(after_suffix_int_i, dim=-2) 
                after_suffix_int.append(after_suffix_int_i)
            output_shape = (bs,) + self.llm_prompt.shape
            after_int = torch.zeros(output_shape).cuda() 
            for i in range(len(self.llm_prompt)): 
                after_int_i = torch.cat([
                    after_prefix_int[:, i].unsqueeze(1),
                    unchanged[i],  
                    after_suffix_int[i]
                ], dim=-2)  
                after_int[:, i, :self.llm_rep_length[i], :] = after_int_i.squeeze(1)
            bs2, n_cls2, len_total2, dim2 = after_int.shape
            assert bs == bs2
            after_int = after_int.view(-1, len_total2, dim2) 
            n_cls, max_len, dim = after_int.shape
            after_int = after_int.view(-1, dim)
            final_output = self.lora_proj(after_int)
            final_output = final_output.reshape(n_cls, max_len, -1)
            final_output_dim = final_output.shape[-1]
            final_output = final_output.view(bs2, n_cls2, len_total2, final_output_dim)
            return final_output

    def forward(self, img_fea):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        llm_prompts = self.forward_llm(img_fea)
        prompts = []
        for b in range(llm_prompts.shape[0]):
            llm_prompts_i = llm_prompts[b]
            prompts_i = self.construct_prompts(ctx, prefix, suffix, llm_prompts_i)
            prompts.append(prompts_i)
        prompts = torch.stack(prompts)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.add_llm_tokenized_prompts = self.prompt_learner.add_llm_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        tokenized_prompts = self.add_llm_tokenized_prompts
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner(image_features) 
        bs, n_cls, len_prompt, dim = prompts.shape
        prompts_reshape = prompts.view(-1, len_prompt, dim) 
        text_features = self.text_encoder(prompts_reshape, tokenized_prompts, image_features) 
        text_features = text_features.reshape(bs, n_cls, text_features.shape[-1])
        logits = []
        for b in range(bs):
            image_features_i = image_features[b].unsqueeze(0)  
            text_features_i = text_features[b] 
            text_features_i = text_features_i / text_features_i.norm(dim=-1, keepdim=True)
            logits_i = logit_scale * image_features_i @ text_features_i.t()
            logits.append(logits_i)  
        logits = torch.cat(logits, dim=0) 
        if self.prompt_learner.training and label is not None:
            fixed_embeddings = None
            with torch.no_grad():
                zero_shot_features = None
                zero_shot_logits = None
            return F.cross_entropy(logits,
                                   label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
        else:
            return logits


@TRAINER_REGISTRY.register()
class CILMP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTSRC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        cfg['classnames'] = classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTSRC.PREC == "fp32" or cfg.TRAINER.PROMPTSRC.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                if "VPT" in name or 'intervention' in name or 'lora_proj' in name: 
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        self.previous_model_gpa = None


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)[0]
            optim.zero_grad()
            loss.backward()
            optim.step()
        loss_summary = {"loss": loss.item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

        if self.step_counter == self.model.total_epochs + 1:
            model.load_state_dict(self.previous_model_gpa)
            self.model.load_state_dict(self.previous_model_gpa)
        return loss_summary


    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                if 'intervention' in key or 'lora_proj' in key:
                    continue
                else:
                    updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                if 'intervention' in key or 'lora_proj' in key:                    
                    modified_dict[key] = dict1[key]
                else:
                    modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

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
        model_file = "model.pth.tar-100"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

