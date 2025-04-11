import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from pycocotools import mask as mask_util
import numpy as np


def decode_mask(mask_encoded):
    """将RLE编码的掩码解码为二维数组
    
    处理json序列化再反序列化的情况
    - mask_encoded可能是带有'counts'和'size'的dict
    - 'counts'可能是str或bytes
    """
    try:
        # 确保mask_encoded是字典格式
        if not isinstance(mask_encoded, dict):
            return None
        
        # 确保包含必要的键
        if 'counts' not in mask_encoded or 'size' not in mask_encoded:
            return None
        
        # 处理counts字段 - 可能是str或bytes
        counts = mask_encoded['counts']
        if isinstance(counts, str):
            # 如果是字符串，转换为bytes
            counts = counts.encode('utf-8')
            mask_encoded['counts'] = counts
        
        # 解码掩码
        mask = mask_util.decode(mask_encoded)
        return mask
    except Exception:
        # 返回空掩码
        return np.zeros((1, 1), dtype=np.uint8)


@LOSS_REGISTRY.register()
class TextSemanticConsistencyLoss(nn.Module):
    """
    语义一致性损失：确保增强特征与文本语义一致
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(TextSemanticConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        
    def forward(self, enhanced_features, text_pooled, feat_proj=None):
        """
        Args:
            enhanced_features (Tensor): 增强后的特征, shape (B, C, H, W)
            text_pooled (Tensor): 文本池化特征, shape (B, D)
            feat_proj (nn.Module, optional): 特征投影层
        """
        # 全局池化特征
        b, c, h, w = enhanced_features.size()
        pooled_features = F.adaptive_avg_pool2d(enhanced_features, (1, 1)).view(b, c)
        
        # 如果需要，投影到相同维度
        if feat_proj is not None:
            pooled_features = feat_proj(pooled_features)
        
        # 计算余弦相似度
        sim = F.cosine_similarity(pooled_features, text_pooled, dim=1)
        
        # 计算损失：最大化相似度，所以使用 1-sim
        loss = 1.0 - sim
        
        # 应用reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class TextRegionAttentionLoss(nn.Module):
    """
    文本区域监督注意力损失：指导注意力集中在与文本相关的对象区域
    """
    def __init__(self, loss_weight=1.0, reduction='mean', entropy_weight=0.05, diversity_weight=0.02):
        super(TextRegionAttentionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # 新增：注意力熵正则化权重 - 防止注意力崩塌
        self.entropy_weight = entropy_weight
        
        # 新增：注意力多样性权重 - 鼓励不同文本产生不同的注意力
        self.diversity_weight = diversity_weight
        
        # 类别同义词映射表，扩展匹配范围
        self.category_synonyms = {
            # 人物相关
            'person': ['person', 'people', 'human', 'man', 'woman', 'child', 'boy', 'girl', 'player', 'pitcher', 'catcher', 'batter', 'individual', 'pedestrian'],
            
            # 交通工具
            'bicycle': ['bicycle', 'bike', 'cycle', 'cycling', 'pedal bike', 'push bike'],
            'car': ['car', 'vehicle', 'automobile', 'sedan', 'suv', 'coupe'],
            'motorcycle': ['motorcycle', 'motorbike', 'bike', 'scooter', 'moped'],
            'airplane': ['airplane', 'plane', 'aircraft', 'jet', 'airliner', 'aviation'],
            'bus': ['bus', 'coach', 'autobus', 'shuttle', 'transit', 'double-decker'],
            'train': ['train', 'locomotive', 'railway', 'subway', 'metro', 'monorail'],
            'truck': ['truck', 'lorry', 'pickup', 'vehicle', 'semi', 'tractor trailer'],
            'boat': ['boat', 'ship', 'vessel', 'sailboat', 'watercraft', 'canoe', 'kayak'],
            
            # 街道物品
            'traffic light': ['traffic light', 'stoplight', 'signal', 'traffic signal', 'semaphore'],
            'fire hydrant': ['fire hydrant', 'hydrant', 'water hydrant', 'fire plug'],
            'stop sign': ['stop sign', 'road sign', 'traffic sign'],
            'parking meter': ['parking meter', 'meter', 'pay station'],
            'bench': ['bench', 'seat', 'park bench', 'outdoor seat'],
            
            # 动物
            'bird': ['bird', 'avian', 'fowl', 'feathered creature', 'winged animal'],
            'cat': ['cat', 'feline', 'kitten', 'kitty', 'tomcat', 'house cat'],
            'dog': ['dog', 'canine', 'puppy', 'hound', 'pooch', 'mutt'],
            'horse': ['horse', 'equine', 'stallion', 'mare', 'pony', 'foal', 'colt'],
            'sheep': ['sheep', 'lamb', 'ewe', 'ram', 'mutton'],
            'cow': ['cow', 'cattle', 'bovine', 'bull', 'calf', 'heifer', 'ox'],
            'elephant': ['elephant', 'pachyderm', 'tusker', 'bull elephant', 'cow elephant'],
            'bear': ['bear', 'grizzly', 'polar bear', 'brown bear', 'black bear', 'cub'],
            'zebra': ['zebra', 'striped horse', 'equid'],
            'giraffe': ['giraffe', 'long-necked animal', 'tall animal'],
            
            # 个人物品
            'backpack': ['backpack', 'rucksack', 'knapsack', 'bag', 'pack', 'schoolbag'],
            'umbrella': ['umbrella', 'parasol', 'bumbershoot', 'rain protection'],
            'handbag': ['handbag', 'purse', 'bag', 'clutch', 'tote', 'satchel'],
            'tie': ['tie', 'necktie', 'bowtie', 'cravat', 'neckwear'],
            'suitcase': ['suitcase', 'luggage', 'travel bag', 'baggage', 'trunk', 'valise'],
            
            # 运动用品
            'frisbee': ['frisbee', 'flying disc', 'disc', 'disk'],
            'skis': ['skis', 'ski', 'snow ski', 'water ski'],
            'snowboard': ['snowboard', 'snow board', 'board', 'winter sports equipment'],
            'sports ball': ['sports ball', 'ball', 'baseball', 'basketball', 'football', 'soccer ball', 'tennis ball', 'volleyball', 'golf ball'],
            'kite': ['kite', 'flying kite', 'wind toy'],
            'baseball bat': ['baseball bat', 'bat', 'club', 'wooden bat'],
            'baseball glove': ['baseball glove', 'glove', 'catcher\'s mitt', 'mitt', 'fielder\'s glove'],
            'skateboard': ['skateboard', 'board', 'deck', 'roller board'],
            'surfboard': ['surfboard', 'surf board', 'board', 'longboard', 'shortboard'],
            'tennis racket': ['tennis racket', 'racket', 'racquet', 'tennis equipment', 'paddle'],
            
            # 食物容器
            'bottle': ['bottle', 'flask', 'container', 'vial', 'water bottle'],
            'wine glass': ['wine glass', 'glass', 'goblet', 'stemware', 'chalice'],
            'cup': ['cup', 'mug', 'tumbler', 'glass', 'teacup', 'coffee cup'],
            'fork': ['fork', 'cutlery', 'silverware', 'utensil', 'dining implement'],
            'knife': ['knife', 'blade', 'cutlery', 'silverware', 'cutting tool', 'utensil'],
            'spoon': ['spoon', 'cutlery', 'silverware', 'utensil', 'dining implement'],
            'bowl': ['bowl', 'dish', 'container', 'basin', 'soup bowl'],
            
            # 食物
            'banana': ['banana', 'fruit', 'plantain', 'yellow fruit'],
            'apple': ['apple', 'fruit', 'red fruit', 'green fruit'],
            'sandwich': ['sandwich', 'sub', 'hoagie', 'panini', 'grilled sandwich'],
            'orange': ['orange', 'fruit', 'citrus', 'citrus fruit'],
            'broccoli': ['broccoli', 'vegetable', 'green vegetable', 'cruciferous vegetable'],
            'carrot': ['carrot', 'vegetable', 'orange vegetable', 'root vegetable'],
            'hot dog': ['hot dog', 'hotdog', 'frankfurter', 'wiener', 'sausage'],
            'pizza': ['pizza', 'pie', 'pizza pie', 'flatbread', 'italian food'],
            'donut': ['donut', 'doughnut', 'pastry', 'sweet treat', 'fried pastry'],
            'cake': ['cake', 'pastry', 'dessert', 'sweet', 'birthday cake', 'cheesecake', 'slice'],
            
            # 家具
            'chair': ['chair', 'seat', 'stool', 'sitting furniture', 'armchair'],
            'couch': ['couch', 'sofa', 'settee', 'loveseat', 'divan', 'futon'],
            'potted plant': ['potted plant', 'plant', 'houseplant', 'flower pot', 'indoor plant'],
            'bed': ['bed', 'mattress', 'bedstead', 'sleeping furniture', 'cot', 'bunk'],
            'dining table': ['dining table', 'table', 'kitchen table', 'dinner table', 'eating surface'],
            'toilet': ['toilet', 'lavatory', 'commode', 'water closet', 'restroom', 'bathroom'],
            
            # 电子产品
            'tv': ['tv', 'television', 'television set', 'monitor', 'screen', 'flat screen'],
            'laptop': ['laptop', 'computer', 'notebook', 'netbook', 'portable computer'],
            'mouse': ['mouse', 'computer mouse', 'pointing device', 'pc mouse'],
            'remote': ['remote', 'remote control', 'controller', 'tv control', 'clicker'],
            'keyboard': ['keyboard', 'computer keyboard', 'typing device', 'input device'],
            'cell phone': ['cell phone', 'mobile phone', 'smartphone', 'phone', 'mobile device', 'iphone'],
            'microwave': ['microwave', 'microwave oven', 'kitchen appliance', 'cooking device'],
            'oven': ['oven', 'stove', 'range', 'baking appliance', 'cooking appliance'],
            'toaster': ['toaster', 'toast maker', 'kitchen appliance', 'bread toaster'],
            'sink': ['sink', 'basin', 'washbasin', 'washbowl', 'kitchen sink', 'bathroom sink'],
            'refrigerator': ['refrigerator', 'fridge', 'freezer', 'cooler', 'icebox', 'cooling appliance'],
            
            # 其他物品
            'book': ['book', 'novel', 'textbook', 'publication', 'volume', 'reading material'],
            'clock': ['clock', 'timepiece', 'wall clock', 'alarm clock', 'time indicator'],
            'vase': ['vase', 'container', 'flower holder', 'decorative vessel', 'urn'],
            'scissors': ['scissors', 'shears', 'cutting tool', 'cutting instrument'],
            'teddy bear': ['teddy bear', 'stuffed animal', 'plush toy', 'teddy', 'stuffed bear'],
            'hair drier': ['hair drier', 'hairdryer', 'hair dryer', 'blow dryer', 'drying appliance'],
            'toothbrush': ['toothbrush', 'dental brush', 'oral hygiene tool', 'dental tool'],
        }
        
    def is_category_in_text(self, category, text):
        """智能检查类别是否在文本中提及
        
        Args:
            category: 对象类别名称
            text: 文本描述
            
        Returns:
            bool: 如果类别或其同义词在文本中，返回True
        """
        # 1. 直接匹配
        if category.lower() in text.lower():
            return True
            
        # 2. 检查同义词 - 直接映射
        if category.lower() in self.category_synonyms:
            synonyms = self.category_synonyms[category.lower()]
            for synonym in synonyms:
                if synonym.lower() in text.lower():
                    return True
        
        # 2.5 检查同义词 - 反向映射（新增）
        # 检查category是否在任何同义词列表中
        category_lower = category.lower()
        for key, synonyms in self.category_synonyms.items():
            if category_lower in [s.lower() for s in synonyms]:
                # 如果category是某个键的同义词，则检查该键和其他同义词是否在文本中
                if key.lower() in text.lower():
                    return True
                    
                for synonym in synonyms:
                    if synonym.lower() != category_lower and synonym.lower() in text.lower():
                        return True
        
        # 3. 针对复合词的部分匹配（如baseball glove -> glove）
        category_parts = category.lower().split()
        if len(category_parts) > 1:
            for part in category_parts:
                if len(part) > 3 and part in text.lower():  # 只匹配长度>3的部分，避免匹配"a"、"the"等
                    return True
        
        return False
        
    def resize_mask(self, mask, target_size):
        """调整掩码大小"""
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
            
        return F.interpolate(mask, size=target_size, mode='nearest')
        
    def entropy_regularization(self, attention_map):
        """
        计算注意力图的熵，鼓励注意力分布更均匀
        
        Args:
            attention_map: 注意力图，形状为 [B, 1, H, W]
            
        Returns:
            entropy_loss: 熵损失，值越低表示熵越高
        """
        # 应用sigmoid获取概率分布
        attention_prob = torch.sigmoid(attention_map)
        
        # 确保数值稳定性
        eps = 1e-7
        attention_prob = attention_prob.clamp(min=eps, max=1.0-eps)
        
        # 计算熵：-p*log(p) - (1-p)*log(1-p)
        entropy = -attention_prob * torch.log(attention_prob) - (1-attention_prob) * torch.log(1-attention_prob)
        
        # 取平均，取负使其成为最小化目标（熵越高越好）
        entropy_loss = -entropy.mean()
        
        return entropy_loss
    
    def diversity_regularization(self, attention_maps, text_prompts):
        """
        鼓励不同文本生成不同的注意力图
        
        Args:
            attention_maps: 批次中的注意力图 [B, 1, H, W]
            text_prompts: 文本提示列表
            
        Returns:
            diversity_loss: 多样性损失
        """
        batch_size = attention_maps.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=attention_maps.device)
        
        # 应用sigmoid获取概率分布
        attention_probs = torch.sigmoid(attention_maps)
        
        # 计算批次内所有注意力图对之间的相似度
        similarities = []
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                # 只在文本不同时计算相似度
                if text_prompts[i] != text_prompts[j]:
                    # 扁平化注意力图
                    attn_i = attention_probs[i].view(-1)
                    attn_j = attention_probs[j].view(-1)
                    
                    # 计算余弦相似度
                    sim = F.cosine_similarity(attn_i.unsqueeze(0), attn_j.unsqueeze(0), dim=1)
                    similarities.append(sim)
        
        # 如果没有不同的文本对，返回零损失
        if not similarities:
            return torch.tensor(0.0, device=attention_maps.device)
        
        # 计算平均相似度，我们希望相似度越低越好
        avg_similarity = torch.stack(similarities).mean()
        
        return avg_similarity  # 直接返回相似度作为损失
        
    def forward(self, attention_maps, text_prompts, objects_info, device=None):
        """
        Args:
            attention_maps (Tensor): 注意力图，shape (B, 1, H, W)
            text_prompts (list): 文本提示列表，长度为B
            objects_info (list): 对象信息列表，每个元素包含一个样本的对象信息
            device (torch.device): 设备
        """
        batch_size = len(text_prompts)
        device = attention_maps.device if device is None else device
        loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        # 新增：计算熵正则化损失
        entropy_loss = self.entropy_regularization(attention_maps)
        
        # 新增：计算多样性正则化损失
        diversity_loss = self.diversity_regularization(attention_maps, text_prompts)
        
        for i in range(batch_size):
            # 获取当前样本的文本和对象信息
            text = text_prompts[i].lower()
            
            if i >= len(objects_info) or not objects_info[i]:
                continue
                
            objects = objects_info[i]
            
            h, w = attention_maps[i].shape[-2:]
            text_region_mask = torch.zeros((1, h, w), device=device)
            found_match = False
            
            # 为文本中提到的对象创建掩码
            for obj in objects:
                if 'category' in obj and 'mask_encoded' in obj:
                    category = obj['category'].lower()
                    
                    # 检查类别是否在文本中提到 - 使用增强的匹配方法
                    if self.is_category_in_text(category, text):
                        found_match = True
                        try:
                            # 解码掩码并调整大小
                            obj_mask = decode_mask(obj['mask_encoded'])
                            if obj_mask is not None:
                                obj_mask = self.resize_mask(obj_mask, (h, w)).to(device)
                                
                                # 合并掩码
                                text_region_mask = torch.max(text_region_mask, obj_mask[0])
                        except Exception:
                            continue
            
            # 如果找到匹配的对象，计算损失
            if found_match:
                attention_map = attention_maps[i:i+1]
                
                sample_loss = self.bce_loss(attention_map, text_region_mask.unsqueeze(0))
                if self.reduction == 'mean':
                    sample_loss = sample_loss.mean()
                elif self.reduction == 'sum':
                    sample_loss = sample_loss.sum()
                
                loss += sample_loss
                valid_samples += 1
        
        # 避免除以零
        if valid_samples > 0:
            loss = loss / valid_samples
            
        # 添加熵正则化和多样性正则化
        final_loss = loss + self.entropy_weight * entropy_loss + self.diversity_weight * diversity_loss
        
        return self.loss_weight * final_loss


@LOSS_REGISTRY.register()
class FeatureRefinementLoss(nn.Module):
    """
    特征细化损失：确保增强过程是有效的，即增强后的特征比原始特征更接近文本语义
    """
    def __init__(self, loss_weight=1.0, margin=0.1, stability_weight=0.1, reduction='mean'):
        super(FeatureRefinementLoss, self).__init__()
        self.loss_weight = loss_weight
        self.margin = margin
        self.stability_weight = stability_weight
        self.reduction = reduction
        
    def forward(self, original_features, enhanced_features, text_pooled, feat_proj=None):
        """
        Args:
            original_features (Tensor): 原始特征，shape (B, C, H, W)
            enhanced_features (Tensor): 增强后的特征，shape (B, C, H, W)
            text_pooled (Tensor): 文本池化特征，shape (B, D)
            feat_proj (nn.Module, optional): 特征投影层
        """
        batch_size = original_features.size(0)
        
        # 全局池化特征
        orig_pooled = F.adaptive_avg_pool2d(original_features, (1, 1)).view(batch_size, -1)
        enhanced_pooled = F.adaptive_avg_pool2d(enhanced_features, (1, 1)).view(batch_size, -1)
        
        # 如果需要，投影到相同维度
        if feat_proj is not None:
            orig_pooled = feat_proj(orig_pooled)
            enhanced_pooled = feat_proj(enhanced_pooled)
        
        # 计算与文本特征的相似度
        orig_sim = F.cosine_similarity(orig_pooled, text_pooled, dim=1)
        enhanced_sim = F.cosine_similarity(enhanced_pooled, text_pooled, dim=1)
        
        # 计算提升损失：确保增强后的相似度更高
        # 使用ReLU确保仅在原始相似度高于增强相似度时有损失
        improvement_loss = F.relu(orig_sim - enhanced_sim + self.margin)
        
        # 稳定性损失：确保增强前后的特征不会相差太大
        stability_loss = F.mse_loss(enhanced_features, original_features, reduction='none')
        
        # 应用reduction
        if self.reduction == 'mean':
            improvement_loss = improvement_loss.mean()
            stability_loss = stability_loss.mean()
        elif self.reduction == 'sum':
            improvement_loss = improvement_loss.sum()
            stability_loss = stability_loss.sum()
            
        # 组合损失
        total_loss = improvement_loss + self.stability_weight * stability_loss
        
        return self.loss_weight * total_loss 