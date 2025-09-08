"""
MAFN-APL: Multi-Scale Attention Fusion Network with Anatomical Prior Learning
File: mafn_apl.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Novel components ----------------
class MorphologicalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MorphologicalConvBlock, self).__init__()

        directional_channels = out_channels // 4
        remaining_channels = out_channels - (3 * directional_channels)

        self.conv_horizontal = nn.Conv2d(in_channels, directional_channels, (1, kernel_size), padding=(0, kernel_size//2))
        self.conv_vertical = nn.Conv2d(in_channels, directional_channels, (kernel_size, 1), padding=(kernel_size//2, 0))
        self.conv_diagonal = nn.Conv2d(in_channels, directional_channels + remaining_channels, kernel_size, padding=kernel_size//2)

        self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.projection = nn.Conv2d(out_channels, out_channels, 1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        h_feat = self.conv_horizontal(x)
        v_feat = self.conv_vertical(x)
        d_feat = self.conv_diagonal(x)

        directional_feat = torch.cat([h_feat, v_feat, d_feat], dim=1)

        edge_feat = self.edge_conv(x)

        if directional_feat.size(1) != edge_feat.size(1):
            directional_feat = self.projection(directional_feat)

        combined = directional_feat + edge_feat
        combined = self.bn(combined)
        combined = self.activation(combined)
        combined = self.dropout(combined)

        return combined


class UncertaintyGuidedAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(UncertaintyGuidedAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2_mean = nn.Linear(channels // reduction, channels)
        self.fc2_logvar = nn.Linear(channels // reduction, channels)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        b, c, h, w = x.size()
        avg_feat = self.avg_pool(x).view(b, c)
        max_feat = self.max_pool(x).view(b, c)
        combined_feat = avg_feat + max_feat

        feat = F.relu(self.fc1(combined_feat))
        feat = self.dropout(feat)

        att_mean = self.fc2_mean(feat)
        att_logvar = self.fc2_logvar(feat)

        attention_weights = self.reparameterize(att_mean, att_logvar)
        attention_weights = self.sigmoid(attention_weights).view(b, c, 1, 1)

        attended_feat = x * attention_weights

        return attended_feat, att_logvar


class MultiScalePyramidAttention(nn.Module):
    def __init__(self, channels):
        super(MultiScalePyramidAttention, self).__init__()

        self.channels = channels
        self.scales = [1, 2, 4, 8]
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1) for _ in self.scales
        ])

        self.cross_scale_attention = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(channels)
        self.fusion_conv = nn.Conv2d(channels * len(self.scales), channels, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        scale_features = []

        for i, (scale, conv) in enumerate(zip(self.scales, self.scale_convs)):
            if scale > 1:
                scaled_x = F.adaptive_avg_pool2d(x, (max(1, h//scale), max(1, w//scale)))
                scaled_feat = conv(scaled_x)
                scaled_feat = F.interpolate(scaled_feat, size=(h, w), mode='bilinear', align_corners=False)
            else:
                scaled_feat = conv(x)
            scale_features.append(scaled_feat)

        stacked_features = torch.stack(scale_features, dim=1)
        stacked_features_reshaped = stacked_features.permute(0, 3, 4, 1, 2)
        stacked_features_reshaped = stacked_features_reshaped.reshape(b * h * w, len(self.scales), c)

        attended_features, _ = self.cross_scale_attention(
            stacked_features_reshaped, stacked_features_reshaped, stacked_features_reshaped
        )
        attended_features = self.layer_norm(attended_features + stacked_features_reshaped)

        attended_features = attended_features.reshape(b, h, w, len(self.scales), c)
        attended_features = attended_features.permute(0, 3, 4, 1, 2)

        scale_features = [attended_features[:, i] for i in range(len(self.scales))]

        fused_features = torch.cat(scale_features, dim=1)
        output = self.fusion_conv(fused_features)

        return output


class AnatomicalRegionDecomposer(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(AnatomicalRegionDecomposer, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Masks sized for typical feature map after initial conv for 224x224 -> 56x56
        self.cortical_mask = nn.Parameter(torch.randn(1, 1, 56, 56))
        self.subcortical_mask = nn.Parameter(torch.randn(1, 1, 56, 56))
        self.ventricular_mask = nn.Parameter(torch.randn(1, 1, 56, 56))

        self.cortical_encoder = self._make_region_encoder(base_channels, base_channels * 2)
        self.subcortical_encoder = self._make_region_encoder(base_channels, base_channels * 2)
        self.ventricular_encoder = self._make_region_encoder(base_channels, base_channels * 2)

    def _make_region_encoder(self, in_channels, out_channels):
        return nn.Sequential(
            MorphologicalConvBlock(in_channels, out_channels),
            MorphologicalConvBlock(out_channels, out_channels),
            MultiScalePyramidAttention(out_channels)
        )

    def forward(self, x):
        base_features = self.initial_conv(x)

        # Resize masks if input size differs
        mask_h, mask_w = self.cortical_mask.shape[2:]
        if base_features.shape[2] != mask_h or base_features.shape[3] != mask_w:
            cortical_mask = F.interpolate(self.cortical_mask, size=base_features.shape[2:], mode='bilinear', align_corners=False)
            subcortical_mask = F.interpolate(self.subcortical_mask, size=base_features.shape[2:], mode='bilinear', align_corners=False)
            ventricular_mask = F.interpolate(self.ventricular_mask, size=base_features.shape[2:], mode='bilinear', align_corners=False)
        else:
            cortical_mask = self.cortical_mask
            subcortical_mask = self.subcortical_mask
            ventricular_mask = self.ventricular_mask

        cortical_feat = base_features * torch.sigmoid(cortical_mask)
        subcortical_feat = base_features * torch.sigmoid(subcortical_mask)
        ventricular_feat = base_features * torch.sigmoid(ventricular_mask)

        cortical_encoded = self.cortical_encoder(cortical_feat)
        subcortical_encoded = self.subcortical_encoder(subcortical_feat)
        ventricular_encoded = self.ventricular_encoder(ventricular_feat)

        return cortical_encoded, subcortical_encoded, ventricular_encoded


class CrossRegionAttentionFusion(nn.Module):
    def __init__(self, channels):
        super(CrossRegionAttentionFusion, self).__init__()

        self.channels = channels
        self.region_attention = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(channels)
        self.fusion_conv = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, cortical, subcortical, ventricular):
        b, c, h, w = cortical.size()

        regions = torch.stack([cortical, subcortical, ventricular], dim=1)

        regions_reshaped = regions.permute(0, 3, 4, 1, 2)
        regions_reshaped = regions_reshaped.reshape(b * h * w, 3, c)

        attended_regions, _ = self.region_attention(regions_reshaped, regions_reshaped, regions_reshaped)
        attended_regions = self.layer_norm(attended_regions + regions_reshaped)

        attended_regions = attended_regions.reshape(b, h, w, 3, c)
        attended_regions = attended_regions.permute(0, 3, 4, 1, 2)

        cortical_att = attended_regions[:, 0]
        subcortical_att = attended_regions[:, 1]
        ventricular_att = attended_regions[:, 2]

        fused = torch.cat([cortical_att, subcortical_att, ventricular_att], dim=1)
        output = self.fusion_conv(fused)

        return output


class TemporalConsistencyModule(nn.Module):
    def __init__(self, channels):
        super(TemporalConsistencyModule, self).__init__()

        self.consistency_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.consistency_attention = UncertaintyGuidedAttention(channels)

    def forward(self, x):
        if self.training:
            augmented = self._random_augment(x)
            consistency_feat = self.consistency_conv(augmented)
            consistency_feat, uncertainty = self.consistency_attention(consistency_feat)
            return x + 0.1 * consistency_feat, uncertainty
        else:
            consistency_feat, uncertainty = self.consistency_attention(x)
            return consistency_feat, uncertainty

    def _random_augment(self, x):
        if torch.rand(1) > 0.5:
            return torch.flip(x, dims=[3])
        return x


class MAFN_APL(nn.Module):
    def __init__(self, num_classes=4, base_channels=64):
        super(MAFN_APL, self).__init__()

        self.anatomical_decomposer = AnatomicalRegionDecomposer(3, base_channels)
        self.cross_region_fusion = CrossRegionAttentionFusion(base_channels * 2)
        self.global_attention = UncertaintyGuidedAttention(base_channels * 2)
        self.temporal_consistency = TemporalConsistencyModule(base_channels * 2)

        self.feature_aggregation = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(base_channels * 2, base_channels),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(base_channels, num_classes)
        )

        self.uncertainties = []

    def forward(self, x):
        self.uncertainties = []

        cortical, subcortical, ventricular = self.anatomical_decomposer(x)

        fused_features = self.cross_region_fusion(cortical, subcortical, ventricular)

        attended_features, global_uncertainty = self.global_attention(fused_features)
        self.uncertainties.append(global_uncertainty)

        consistent_features, temporal_uncertainty = self.temporal_consistency(attended_features)
        self.uncertainties.append(temporal_uncertainty)

        aggregated_features = self.feature_aggregation(consistent_features)
        output = self.classifier(aggregated_features)

        return output


# ---------------- Composite Loss ----------------
class CompositeLoss(nn.Module):
    def __init__(self, num_classes=4):
        super(CompositeLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def anatomical_consistency_loss(self, model, features=None):
        cortical_mask = torch.sigmoid(model.anatomical_decomposer.cortical_mask)
        subcortical_mask = torch.sigmoid(model.anatomical_decomposer.subcortical_mask)
        ventricular_mask = torch.sigmoid(model.anatomical_decomposer.ventricular_mask)

        smooth_loss = 0
        for mask in [cortical_mask, subcortical_mask, ventricular_mask]:
            dx = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
            dy = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
            smooth_loss += torch.mean(dx) + torch.mean(dy)

        return smooth_loss

    def uncertainty_regularization_loss(self, uncertainties):
        if not uncertainties:
            return torch.tensor(0.0, device='cpu')

        uncertainty_loss = 0
        for uncertainty in uncertainties:
            uncertainty_loss += torch.mean(torch.abs(uncertainty - 0.5))

        return uncertainty_loss / len(uncertainties)

    def forward(self, outputs, targets, model, original_features=None, consistent_features=None):
        cls_loss = self.classification_loss(outputs, targets)
        anat_loss = self.anatomical_consistency_loss(model, None)
        uncert_loss = self.uncertainty_regularization_loss(model.uncertainties)

        total_loss = (cls_loss +
                     0.1 * anat_loss +
                     0.05 * uncert_loss)

        return total_loss, {
            'classification': cls_loss.item(),
            'anatomical': anat_loss.item(),
            'uncertainty': uncert_loss.item()
        }


# ---------------- Quick sanity check ----------------
if __name__ == "__main__":
    model = MAFN_APL(num_classes=4, base_channels=32)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)
