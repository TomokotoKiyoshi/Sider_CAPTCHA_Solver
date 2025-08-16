#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练日志解析器 - 从日志文件中提取指标并生成metrics_history.json
"""
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime


class TrainingLogParser:
    """
    训练日志解析器
    
    功能：
    1. 解析训练日志文件
    2. 提取每个epoch的训练和验证指标
    3. 生成结构化的metrics_history.json
    4. 支持多种日志格式
    """
    
    def __init__(self):
        """初始化日志解析器"""
        self.logger = logging.getLogger('LogParser')
        
        # 定义正则表达式模式
        self.patterns = {
            # Epoch信息: "Epoch 1/40"
            'epoch_info': r'Epoch\s+(\d+)/(\d+)',
            
            # 学习率: "学习率: 0.000300"
            'learning_rate': r'学习率:\s*([\d.]+)',
            
            # 训练损失相关
            'train_loss': r'Loss:\s*([\d.]+)',
            'focal_loss': r'Focal:\s*([\d.]+)',
            'offset_loss': r'Offset:\s*([\d.]+)',
            
            # 验证指标
            'val_mae': r'验证MAE:\s*([\d.]+)px',
            'val_hit2': r'验证Hit@2px:\s*([\d.]+)%',
            'val_hit5': r'验证Hit@5px:\s*([\d.]+)%',
            
            # 详细验证指标（从验证部分提取）
            'gap_mae': r'Gap MAE:\s*([\d.]+)\s*px',
            'slider_mae': r'Slider MAE:\s*([\d.]+)\s*px',
            'val_rmse': r'RMSE:\s*([\d.]+)\s*px',
            'val_hit1': r'Hit@1px:\s*([\d.]+)%',
            'val_hit10': r'Hit@10px:\s*([\d.]+)%',
            
            # 最佳模型标记
            'best_model': r'保存最佳模型',
            'early_stop': r'触发早停',
            
            # 时间信息
            'epoch_time': r'Epoch\s+\d+\s+用时:\s*([\d.]+)秒',
            'eta': r'预计剩余时间:\s*([:\d]+)'
        }
        
        # 存储解析结果
        self.train_history = {}
        self.val_history = {}
        self.meta_info = {}
        
    def parse_log_file(self, log_path: str) -> Dict:
        """
        解析日志文件
        
        Args:
            log_path: 日志文件路径
            
        Returns:
            包含所有提取指标的字典
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"日志文件不存在: {log_path}")
        
        self.logger.info(f"开始解析日志文件: {log_path}")
        
        # 读取日志内容
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 按行分割以便逐epoch处理
        lines = log_content.split('\n')
        
        # 解析日志
        current_epoch = 0
        current_train_metrics = {}
        current_val_metrics = {}
        total_epochs = 0
        best_epoch = None
        
        for i, line in enumerate(lines):
            # 检测Epoch开始
            epoch_match = re.search(self.patterns['epoch_info'], line)
            if epoch_match:
                # 保存上一个epoch的数据（如果有）
                if current_epoch > 0:
                    if current_train_metrics:
                        self.train_history[str(current_epoch)] = current_train_metrics
                    if current_val_metrics:
                        self.val_history[str(current_epoch)] = current_val_metrics
                
                # 开始新的epoch
                current_epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                current_train_metrics = {'epoch': current_epoch}
                current_val_metrics = {'epoch': current_epoch}
                
                self.logger.debug(f"解析 Epoch {current_epoch}/{total_epochs}")
            
            # 提取学习率
            lr_match = re.search(self.patterns['learning_rate'], line)
            if lr_match and current_epoch > 0:
                current_train_metrics['lr'] = float(lr_match.group(1))
            
            # 提取训练损失
            if 'Loss:' in line and current_epoch > 0:
                loss_match = re.search(self.patterns['train_loss'], line)
                if loss_match:
                    current_train_metrics['loss'] = float(loss_match.group(1))
                
                focal_match = re.search(self.patterns['focal_loss'], line)
                if focal_match:
                    current_train_metrics['focal_loss'] = float(focal_match.group(1))
                
                offset_match = re.search(self.patterns['offset_loss'], line)
                if offset_match:
                    current_train_metrics['offset_loss'] = float(offset_match.group(1))
            
            # 提取验证指标（简要版本）
            if '验证MAE:' in line and current_epoch > 0:
                mae_match = re.search(self.patterns['val_mae'], line)
                if mae_match:
                    current_val_metrics['mae_px'] = float(mae_match.group(1))
                
                hit2_match = re.search(self.patterns['val_hit2'], line)
                if hit2_match:
                    current_val_metrics['hit_le_2px'] = float(hit2_match.group(1))
                
                hit5_match = re.search(self.patterns['val_hit5'], line)
                if hit5_match:
                    current_val_metrics['hit_le_5px'] = float(hit5_match.group(1))
            
            # 提取详细验证指标
            if 'Gap MAE:' in line and current_epoch > 0:
                gap_mae_match = re.search(self.patterns['gap_mae'], line)
                if gap_mae_match:
                    current_val_metrics['gap_mae'] = float(gap_mae_match.group(1))
            
            if 'Slider MAE:' in line and current_epoch > 0:
                slider_mae_match = re.search(self.patterns['slider_mae'], line)
                if slider_mae_match:
                    current_val_metrics['slider_mae'] = float(slider_mae_match.group(1))
            
            if 'RMSE:' in line and current_epoch > 0:
                rmse_match = re.search(self.patterns['val_rmse'], line)
                if rmse_match:
                    current_val_metrics['rmse_px'] = float(rmse_match.group(1))
            
            if 'Hit@1px:' in line and current_epoch > 0:
                hit1_match = re.search(self.patterns['val_hit1'], line)
                if hit1_match:
                    current_val_metrics['hit_le_1px'] = float(hit1_match.group(1))
            
            if 'Hit@10px:' in line and current_epoch > 0:
                hit10_match = re.search(self.patterns['val_hit10'], line)
                if hit10_match:
                    current_val_metrics['hit_le_10px'] = float(hit10_match.group(1))
            
            # 检测最佳模型
            if '保存最佳模型' in line and current_epoch > 0:
                current_val_metrics['is_best'] = True
                best_epoch = current_epoch
            
            # 检测早停
            if '触发早停' in line and current_epoch > 0:
                current_val_metrics['early_stop'] = True
            
            # 提取时间信息
            time_match = re.search(self.patterns['epoch_time'], line)
            if time_match and current_epoch > 0:
                current_val_metrics['epoch_time'] = float(time_match.group(1))
        
        # 保存最后一个epoch的数据
        if current_epoch > 0:
            if current_train_metrics:
                self.train_history[str(current_epoch)] = current_train_metrics
            if current_val_metrics:
                self.val_history[str(current_epoch)] = current_val_metrics
        
        # 提取最终指标（训练完成部分）
        best_metrics = self._extract_final_metrics(log_content)
        
        # 构建meta信息
        self.meta_info = {
            'log_file': str(log_path),
            'parse_time': datetime.now().isoformat(),
            'total_epochs': total_epochs,
            'actual_epochs': len(self.val_history),
            'best_epoch': best_epoch,
            'best_metrics': best_metrics,
            'early_stopped': any(m.get('early_stop', False) for m in self.val_history.values())
        }
        
        self.logger.info(f"解析完成: 找到 {len(self.val_history)} 个epoch的数据")
        
        return {
            'meta': self.meta_info,
            'train': self.train_history,
            'validation': self.val_history
        }
    
    def _extract_final_metrics(self, log_content: str) -> Optional[Dict]:
        """
        提取最终/最佳指标
        
        从日志的"最佳模型"部分提取
        """
        # 查找最佳模型部分
        best_pattern = r'最佳模型.*?Epoch\s+(\d+).*?MAE:\s*([\d.]+)px.*?RMSE:\s*([\d.]+)px.*?Hit@2px:\s*([\d.]+)%.*?Hit@5px:\s*([\d.]+)%'
        
        match = re.search(best_pattern, log_content, re.DOTALL)
        if match:
            return {
                'epoch': int(match.group(1)),
                'mae_px': float(match.group(2)),
                'rmse_px': float(match.group(3)),
                'hit_le_2px': float(match.group(4)),
                'hit_le_5px': float(match.group(5))
            }
        
        return None
    
    def save_metrics_history(self, output_dir: str):
        """
        保存解析结果为metrics_history.json
        
        Args:
            output_dir: 输出目录（通常是checkpoint目录）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'metrics_history.json'
        
        # 组合所有数据
        full_data = {
            'meta': self.meta_info,
            'train': self.train_history,
            'validation': self.val_history
        }
        
        # 保存JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"指标历史已保存到: {output_file}")
        
        # 同时生成简化版本（兼容旧格式）
        simple_file = output_dir / 'metrics_history_simple.json'
        simple_data = {}
        
        for epoch_str, metrics in self.val_history.items():
            simple_data[epoch_str] = {
                'mae_px': metrics.get('mae_px', 0),
                'rmse_px': metrics.get('rmse_px', 0),
                'gap_mae': metrics.get('gap_mae', 0),
                'slider_mae': metrics.get('slider_mae', 0),
                'hit_le_1px': metrics.get('hit_le_1px', 0),
                'hit_le_2px': metrics.get('hit_le_2px', 0),
                'hit_le_5px': metrics.get('hit_le_5px', 0),
                'hit_le_10px': metrics.get('hit_le_10px', 0),
                'is_best': metrics.get('is_best', False),
                'early_stop': metrics.get('early_stop', False)
            }
        
        with open(simple_file, 'w', encoding='utf-8') as f:
            json.dump(simple_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"简化版指标历史已保存到: {simple_file}")
        
        return output_file
    
    def print_summary(self):
        """打印解析摘要"""
        if not self.val_history:
            print("没有解析到验证数据")
            return
        
        print("\n" + "="*60)
        print("日志解析摘要")
        print("="*60)
        
        print(f"总Epochs: {self.meta_info.get('total_epochs', 'N/A')}")
        print(f"实际训练Epochs: {self.meta_info.get('actual_epochs', 'N/A')}")
        print(f"早停: {'是' if self.meta_info.get('early_stopped', False) else '否'}")
        
        if self.meta_info.get('best_metrics'):
            best = self.meta_info['best_metrics']
            print(f"\n最佳模型 (Epoch {best.get('epoch', 'N/A')}):")
            print(f"  MAE: {best.get('mae_px', 'N/A'):.3f} px")
            print(f"  RMSE: {best.get('rmse_px', 'N/A'):.3f} px")
            print(f"  Hit@2px: {best.get('hit_le_2px', 'N/A'):.2f}%")
            print(f"  Hit@5px: {best.get('hit_le_5px', 'N/A'):.2f}%")
        
        print("="*60)


def main():
    """主函数 - 命令行使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='解析训练日志并生成metrics_history.json')
    parser.add_argument('log_file', help='日志文件路径')
    parser.add_argument('--output', '-o', default='src/checkpoints/1.1.0', 
                       help='输出目录（默认: src/checkpoints/1.1.0）')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    # 创建解析器
    parser = TrainingLogParser()
    
    # 解析日志
    try:
        data = parser.parse_log_file(args.log_file)
        
        # 保存结果
        output_file = parser.save_metrics_history(args.output)
        
        # 打印摘要
        parser.print_summary()
        
        print(f"\n✅ 解析完成！文件已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n❌ 解析失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())