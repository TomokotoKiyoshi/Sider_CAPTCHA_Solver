# -*- coding: utf-8 -*-
"""
数据分割配置 - 按原始图片分割，避免数据泄漏
"""
import json
import random
from pathlib import Path
from typing import Dict, List


class DataSplitConfig:
    """数据分割配置 - 简化版"""
    
    def __init__(self):
        # 分割比例
        self.train_ratio = 0.8      # 训练集 80%
        self.val_ratio = 0.1        # 验证集 10%  
        self.test_ratio = 0.1       # 测试集 10%
        
        # 随机种子（确保可重现）
        self.seed = 42
        self.shuffle = True
        
        # 数据目录
        self.source_dir = "data/captchas"  # 实际的验证码目录
        self.output_dir = "data/splits"    # 分割结果输出目录
        self.output_file = "data/splits/data_split.json"  # 分割结果文件
    
    def split_by_image(self, file_list: List[str]) -> Dict[str, List[str]]:
        """
        按原始图片ID分割数据，确保同一张图的所有变体在同一集合
        避免数据泄漏：训练时看过Pic0001，测试时就不应该再看到Pic0001的任何变体
        
        Args:
            file_list: 所有图片文件路径列表
            
        Returns:
            包含验证码和组件的完整分割结果
        """
        # 1. 按图片ID分组
        image_groups = {}
        for filepath in file_list:
            # 提取图片ID (如 Pic0001)
            filename = Path(filepath).stem
            pic_id = filename.split('_')[0]  # 获取 Pic0001 部分
            
            if pic_id not in image_groups:
                image_groups[pic_id] = []
            image_groups[pic_id].append(filepath)
        
        # 2. 获取所有唯一的图片ID
        unique_pic_ids = list(image_groups.keys())
        print(f"发现 {len(unique_pic_ids)} 张原始图片")
        
        # 3. 设置随机种子并打乱
        random.seed(self.seed)
        if self.shuffle:
            random.shuffle(unique_pic_ids)
        
        # 4. 计算分割点
        n_pics = len(unique_pic_ids)
        train_end = int(n_pics * self.train_ratio)
        val_end = train_end + int(n_pics * self.val_ratio)
        
        # 5. 分割图片ID
        train_pic_ids = unique_pic_ids[:train_end]
        val_pic_ids = unique_pic_ids[train_end:val_end]
        test_pic_ids = unique_pic_ids[val_end:]
        
        # 6. 收集每个集合的所有文件（包括组件）
        train_captchas = []
        train_sliders = []
        train_backgrounds = []
        train_sample_ids = []
        val_captchas = []
        val_sliders = []
        val_backgrounds = []
        val_sample_ids = []
        test_captchas = []
        test_sliders = []
        test_backgrounds = []
        test_sample_ids = []
        
        # 为每个分割集合收集文件
        for pic_id in train_pic_ids:
            for captcha_path in image_groups[pic_id]:
                train_captchas.append(captcha_path)
                # 根据验证码文件名生成组件文件名
                filename = Path(captcha_path).stem
                train_sample_ids.append(filename)  # 样本ID就是文件名（不含扩展名）
                slider_path = f"data/components/sliders/{filename}_slider.png"
                bg_path = f"data/components/backgrounds/{filename}_gap.png"
                train_sliders.append(slider_path)
                train_backgrounds.append(bg_path)
        
        for pic_id in val_pic_ids:
            for captcha_path in image_groups[pic_id]:
                val_captchas.append(captcha_path)
                filename = Path(captcha_path).stem
                val_sample_ids.append(filename)
                slider_path = f"data/components/sliders/{filename}_slider.png"
                bg_path = f"data/components/backgrounds/{filename}_gap.png"
                val_sliders.append(slider_path)
                val_backgrounds.append(bg_path)
        
        for pic_id in test_pic_ids:
            for captcha_path in image_groups[pic_id]:
                test_captchas.append(captcha_path)
                filename = Path(captcha_path).stem
                test_sample_ids.append(filename)
                slider_path = f"data/components/sliders/{filename}_slider.png"
                bg_path = f"data/components/backgrounds/{filename}_gap.png"
                test_sliders.append(slider_path)
                test_backgrounds.append(bg_path)
        
        # 7. 打印统计信息
        print(f"\n数据分割结果:")
        print(f"训练集: {len(train_pic_ids):4d} 张图片 -> {len(train_captchas):6d} 个样本")
        print(f"验证集: {len(val_pic_ids):4d} 张图片 -> {len(val_captchas):6d} 个样本")
        print(f"测试集: {len(test_pic_ids):4d} 张图片 -> {len(test_captchas):6d} 个样本")
        print(f"总计:   {len(unique_pic_ids):4d} 张图片 -> {len(file_list):6d} 个样本")
        
        # 验证没有数据泄漏
        self._verify_no_leak(train_pic_ids, val_pic_ids, test_pic_ids)
        
        return {
            "train": {
                "captchas": train_captchas,
                "sliders": train_sliders,
                "backgrounds": train_backgrounds,
                "sample_ids": train_sample_ids
            },
            "val": {
                "captchas": val_captchas,
                "sliders": val_sliders,
                "backgrounds": val_backgrounds,
                "sample_ids": val_sample_ids
            },
            "test": {
                "captchas": test_captchas,
                "sliders": test_sliders,
                "backgrounds": test_backgrounds,
                "sample_ids": test_sample_ids
            },
            # 保留原始的pic_id分组信息
            "pic_ids": {
                "train": train_pic_ids,
                "val": val_pic_ids,
                "test": test_pic_ids
            },
            # 添加样本ID映射，便于查找标签
            "sample_ids": {
                "train": train_sample_ids,
                "val": val_sample_ids,
                "test": test_sample_ids
            }
        }
    
    def _verify_no_leak(self, train_ids, val_ids, test_ids):
        """验证数据集之间没有重叠（无数据泄漏）"""
        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)
        
        # 检查是否有交集
        if train_set & val_set:
            raise ValueError("训练集和验证集有重叠！")
        if train_set & test_set:
            raise ValueError("训练集和测试集有重叠！")
        if val_set & test_set:
            raise ValueError("验证集和测试集有重叠！")
        
        print("✓ 数据集无重叠，无数据泄漏风险")
    
    def save_split_result(self, split_result: Dict, output_path: str):
        """保存分割结果到JSON文件"""
        output = {
            "config": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "seed": self.seed,
                "labels_file": "data/labels/all_labels.json"  # 标签文件路径
            },
            "splits": {
                "train": split_result["train"],
                "val": split_result["val"],
                "test": split_result["test"]
            },
            "pic_ids": split_result["pic_ids"],
            "sample_ids": split_result.get("sample_ids", {}),  # 样本ID映射
            "statistics": {
                "train": {
                    "pic_count": len(split_result["pic_ids"]["train"]),
                    "captcha_count": len(split_result["train"]["captchas"]),
                    "slider_count": len(split_result["train"]["sliders"]),
                    "background_count": len(split_result["train"]["backgrounds"])
                },
                "val": {
                    "pic_count": len(split_result["pic_ids"]["val"]),
                    "captcha_count": len(split_result["val"]["captchas"]),
                    "slider_count": len(split_result["val"]["sliders"]),
                    "background_count": len(split_result["val"]["backgrounds"])
                },
                "test": {
                    "pic_count": len(split_result["pic_ids"]["test"]),
                    "captcha_count": len(split_result["test"]["captchas"]),
                    "slider_count": len(split_result["test"]["sliders"]),
                    "background_count": len(split_result["test"]["backgrounds"])
                },
                "total": {
                    "pic_count": sum(len(split_result["pic_ids"][k]) for k in ["train", "val", "test"]),
                    "sample_count": sum(len(split_result[k]["captchas"]) for k in ["train", "val", "test"])
                }
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n分割结果已保存到: {output_path}")


