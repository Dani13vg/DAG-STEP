import numpy as np
import torch
import os
import math
from torch import nn
import torch.nn.functional as F

from adet.layers.deformable_transformer import DeformableTransformer

from adet.layers.pos_encoding import PositionalEncoding1D
from adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset
from adet.utils.queries import max_query_types
from adet.utils.tokenizer import train_tokenizer


# Tokenizer path
tokenizer_path = "tokenizers/default_tokenizer.json"

# -------------------------------------------------------------------------
# Positional Encoding
# -------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe = torch.zeros(max_len, 1, d_model)  # [max_len, 1, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # sin to even indices
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # cos to odd indices
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor, shape [seq_len, batch_size, d_model] OR [batch_size, seq_len, d_model].
           We will add the positional encodings to the first dimension if x is [seq_len, batch_size, d_model].
        """
        # print("x shape:", x.shape) # [seq_len, batch_size, d_model]
        # print("Expected shape:", self.pe.size()) # [max_len, 1, d_model] max_len = 5000
        if x.dim() == 3 and x.size(0) <= self.pe.size(0):
            # shape [seq_len, batch_size, d_model]
            x = x + self.pe[: x.size(0)]
        elif x.dim() == 3 and x.size(1) <= self.pe.size(0):
            # shape [batch_size, seq_len, d_model], we might want to transpose first
            # or simply apply the position along dim=1. Let's do an example approach:
            # If you prefer [batch_size, seq_len, d_model], you can transpose first, apply, transpose back
            x = x.transpose(0, 1)  # => [seq_len, batch_size, d_model]
            x = x + self.pe[: x.size(0)]
            x = x.transpose(0, 1)  # => [batch_size, seq_len, d_model]
        else:
            raise ValueError("Positional encoding: input is too large or shape not supported.")
        return self.dropout(x)

# -------------------------------------------------------------------------
# Regex Encoder
# -------------------------------------------------------------------------
class RegexEncoder(nn.Module):
    """
    Encodes a sequence of regex tokens using a Transformer encoder,
    returning shape [seq_len, batch_size, d_model].
    """
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, dropout=0.5):
        super(RegexEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model, padding_idx=1)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, src):
        """
        src: [regex_seq_len, batch_size] -- typically you might pass it transposed.
        returns: [regex_seq_len, batch_size, d_model]
        """
        # If src is [batch_size, seq_len], you may need to transpose to [seq_len, batch_size].
        # Let's assume the caller ensures shape [seq_len, batch_size].
        embedded = self.embedding(src) * math.sqrt(self.d_model)  # [seq_len, batch_size, d_model]
        embedded = self.pos_encoder(embedded)                     # add positional encoding
        output = self.transformer_encoder(embedded)               # [seq_len, batch_size, d_model]
        return output
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class STEP(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone

        # fmt: off
        self.d_model                 = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead                   = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers      = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers      = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward         = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout                 = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation              = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels      = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points            = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points            = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals           = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale         = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points         = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.num_classes             = 1
        self.max_text_len            = cfg.MODEL.TRANSFORMER.NUM_CHARS
        self.voc_size                = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.sigmoid_offset          = not cfg.MODEL.TRANSFORMER.USE_POLYGON
        self.encoder_weights_path    = cfg.MODEL.TRANSFORMER.ENCODER_WEIGHTS

        self.text_pos_embed = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        self.query_pos_embed = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        # fmt: on

        self.transformer = DeformableTransformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points, num_proposals=self.num_proposals,
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.text_global_class = nn.Linear(self.d_model, self.num_classes)  # text/no text
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)
        self.text_class = nn.Linear(self.d_model, self.voc_size + 1)
        # self.query_embed = MLP(max_query_types(), self.d_model, self.d_model, 2)

        # Regex layers
        self.tokenizer = train_tokenizer(content=None, save_path=tokenizer_path) # Just loads the tokenizer if given a path
        self.encoder_dim = cfg.MODEL.TRANSFORMER.ENCODER_DIM
        
        self.query_embed = RegexEncoder(input_dim=self.tokenizer.get_vocab_size(), d_model=self.encoder_dim, nhead=8,
                                        num_layers=4, dropout=0.4)#.to(self.device)

        # Load pre-trained weights for the regex encoder
        if self.encoder_weights_path:
            if not os.path.exists(self.encoder_weights_path):
                raise FileNotFoundError(f"Encoder weights file not found at: {self.encoder_weights_path}")
            encoder_state_dict = torch.load(self.encoder_weights_path, map_location=self.device)
            self.query_embed.load_state_dict(encoder_state_dict, strict=False)
            print(f"Regex encoder loaded from pretrained weights: {self.encoder_weights_path}")
        else:
            print("No encoder weights provided, using randomly initialized Regex encoder.")

        # Define a linear layer to project the regex encoder output to the transformer input size [from 512 to 256]
        if self.encoder_dim != self.d_model:
            self.query_reprojection = nn.Linear(self.encoder_dim, self.d_model)

        # shared prior between instances (objects)
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)
        self.text_embed = nn.Embedding(self.max_text_len, self.d_model)

        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model,
                              kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.d_model),
                ))
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )])
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.text_global_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        self.text_global_class = nn.ModuleList(
            [self.text_global_class for _ in range(num_pred)])
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.to(self.device)

    def forward(self, x: tuple):
        """ The forward expects a Tuple, which consists of:
                A NestedTensor, which consists of:
                   - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
                   - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
                A list of queries, one one-hot encoded tensor per image as a Tensor of shape [batch_size x num_words
                x num_query_types]
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_keypoints": The normalized keypoint coordinates for all queries, represented as
                               (x, y). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        samples = x[0]
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        if self.num_feature_levels == 1:
            features = [features[-1]]
            pos = [pos[-1]]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                mask = F.interpolate(
                    m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # n_points, embed_dim --> n_objects, n_points, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)
        
        text_pos_embed = self.text_pos_embed(self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        # Reshape input to be [seq_len, batch_size]
        # x[1] = x[1].squeeze(0).transpose(0, 1)
        if type(x[1]) == list:
            x = (x[0], x[1][0])
        x[1].to(self.device)
        queries = self.query_embed(x[1].squeeze(1).transpose(0, 1).long().to(self.device))
        if self.encoder_dim != self.d_model:
            queries = self.query_reprojection(queries)
        queries = queries.transpose(0, 1)
        queries = queries.unsqueeze(1).repeat(1, self.num_proposals, 1, 1)
        query_pos_embed = None #self.query_pos_embed(queries[0, 0]).unsqueeze(0).repeat(queries.shape[0], self.num_proposals, 1, 1) # This may be not needed because we are using the regex encoder with positional encoding

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, queries, query_pos_embed, text_mask=None)

        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                tmp += reference[:, :, None, :2]
            outputs_texts.append(self.text_class(hs_text[lvl]))
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_coords.append(outputs_coord)
            outputs_class_text = self.text_global_class[lvl](hs_text[lvl])
            outputs_classes.append(torch.cat((outputs_class_text, outputs_class), dim=2))
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_text = torch.stack(outputs_texts)

        out = {'pred_logits': outputs_class[-1],
               'pred_ctrl_points': outputs_coord[-1],
               'pred_texts': outputs_text[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_text)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_ctrl_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]
