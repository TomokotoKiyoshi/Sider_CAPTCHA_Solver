# -*- coding: utf-8 -*-
"""
测试LiteBlock的Project层是否为线性（无激活函数）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from src.models.modules import LiteBlock


def test_liteblock_linear_project():
    """验证LiteBlock的Project层是线性的"""
    print("=" * 60)
    print("测试LiteBlock Project层的线性特性")
    print("=" * 60)
    
    # 创建LiteBlock
    in_channels = 32
    out_channels = 32
    expansion = 2
    
    block = LiteBlock(in_channels, out_channels, expansion)
    block.eval()
    
    # 打印模块结构
    print("\nLiteBlock结构:")
    print(f"  输入通道: {in_channels}")
    print(f"  输出通道: {out_channels}")
    print(f"  扩张倍率: {expansion}")
    print(f"  隐藏通道: {in_channels * expansion}")
    
    # 检查各层的激活函数设置
    print("\n激活函数配置检查:")
    
    # 1. PW-Expand层
    print("\n1. PW-Expand层:")
    has_silu_expand = any(isinstance(m, nn.SiLU) for m in block.pw_expand.modules())
    has_identity_expand = any(isinstance(m, nn.Identity) for m in block.pw_expand.modules())
    print(f"   包含SiLU激活: {has_silu_expand}")
    print(f"   包含Identity(无激活): {has_identity_expand}")
    assert has_silu_expand and not has_identity_expand, "PW-Expand应该有SiLU激活"
    
    # 2. DW-Conv层
    print("\n2. DW-Conv层:")
    has_silu_dw = any(isinstance(m, nn.SiLU) for m in block.dw_conv.modules())
    has_identity_dw = any(isinstance(m, nn.Identity) for m in block.dw_conv.modules())
    print(f"   包含SiLU激活: {has_silu_dw}")
    print(f"   包含Identity(无激活): {has_identity_dw}")
    assert has_silu_dw and not has_identity_dw, "DW-Conv应该有SiLU激活"
    
    # 3. PW-Project层（关键：应该是线性的）
    print("\n3. PW-Project层（应为线性）:")
    has_silu_project = any(isinstance(m, nn.SiLU) for m in block.pw_project.modules())
    has_identity_project = any(isinstance(m, nn.Identity) for m in block.pw_project.modules())
    print(f"   包含SiLU激活: {has_silu_project}")
    print(f"   包含Identity(无激活): {has_identity_project}")
    assert not has_silu_project and has_identity_project, "PW-Project应该是线性的（Identity激活）"
    
    # 验证Project层只包含Conv2d、BatchNorm2d和Identity
    project_modules = list(block.pw_project.modules())
    module_types = [type(m).__name__ for m in project_modules]
    print(f"   包含的模块类型: {set(module_types)}")
    
    expected_types = {'ConvBNAct', 'Sequential', 'Conv2d', 'BatchNorm2d', 'Identity'}
    actual_types = set(module_types)
    assert actual_types.issubset(expected_types), f"Project层包含意外的模块类型: {actual_types - expected_types}"
    
    print("\n✓ PW-Project层确认为线性（仅BN，无激活）")
    
    # 功能测试：验证线性特性
    print("\n功能测试：验证线性叠加特性")
    
    # 创建测试输入
    x1 = torch.randn(1, in_channels, 16, 16)
    x2 = torch.randn(1, in_channels, 16, 16)
    alpha = 0.5
    
    # 单独处理Project层
    with torch.no_grad():
        # 先通过expand和dw层
        hidden1 = block.dw_conv(block.pw_expand(x1))
        hidden2 = block.dw_conv(block.pw_expand(x2))
        
        # 线性组合后通过project
        hidden_combined = alpha * hidden1 + (1 - alpha) * hidden2
        out_combined = block.pw_project(hidden_combined)
        
        # 分别通过project后线性组合
        out1 = block.pw_project(hidden1)
        out2 = block.pw_project(hidden2)
        out_linear = alpha * out1 + (1 - alpha) * out2
        
        # 如果Project是线性的，两者应该相等（考虑数值误差）
        diff = torch.abs(out_combined - out_linear).max().item()
        print(f"  线性叠加误差: {diff:.2e}")
        
        # BatchNorm在eval模式下是线性的，所以误差应该很小
        assert diff < 1e-5, f"Project层不满足线性特性，误差: {diff}"
    
    print("  ✓ Project层满足线性叠加特性")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！LiteBlock的Project层确认为线性输出")
    print("=" * 60)


def test_all_stages_liteblock():
    """测试所有Stage中的LiteBlock配置"""
    print("\n" + "=" * 60)
    print("测试所有Stage中的LiteBlock配置")
    print("=" * 60)
    
    from src.models.stage2 import Stage2
    from src.models.stage3 import Stage3
    from src.models.stage4 import Stage4
    from src.models.config_loader import get_stage2_config, get_stage3_config, get_stage4_config
    
    # 测试Stage2
    print("\nStage2中的LiteBlock:")
    stage2_config = get_stage2_config()
    stage2 = Stage2(
        in_channels=stage2_config['in_channels'],
        channels=stage2_config['channels'],
        num_blocks=stage2_config['num_blocks'],
        expansion=stage2_config['expansion']
    )
    
    # 检查Stage2中的所有LiteBlock
    liteblocks_count = 0
    for name, module in stage2.named_modules():
        if isinstance(module, LiteBlock):
            liteblocks_count += 1
            # 验证Project层是线性的
            has_silu = any(isinstance(m, nn.SiLU) for m in module.pw_project.modules())
            assert not has_silu, f"Stage2中的{name}.pw_project不应该有激活函数"
    
    print(f"  找到 {liteblocks_count} 个LiteBlock")
    print(f"  ✓ 所有LiteBlock的Project层都是线性的")
    
    # 测试Stage3
    print("\nStage3中的LiteBlock:")
    stage3_config = get_stage3_config()
    stage3 = Stage3(
        channels=stage3_config['channels'],
        num_blocks=stage3_config['num_blocks'],
        expansion=stage3_config['expansion']
    )
    
    liteblocks_count = 0
    for name, module in stage3.named_modules():
        if isinstance(module, LiteBlock):
            liteblocks_count += 1
            has_silu = any(isinstance(m, nn.SiLU) for m in module.pw_project.modules())
            assert not has_silu, f"Stage3中的{name}.pw_project不应该有激活函数"
    
    print(f"  找到 {liteblocks_count} 个LiteBlock")
    print(f"  ✓ 所有LiteBlock的Project层都是线性的")
    
    # 测试Stage4
    print("\nStage4中的LiteBlock:")
    stage4_config = get_stage4_config()
    stage4 = Stage4(
        channels=stage4_config['channels'],
        num_blocks=stage4_config['num_blocks'],
        expansion=stage4_config['expansion']
    )
    
    liteblocks_count = 0
    for name, module in stage4.named_modules():
        if isinstance(module, LiteBlock):
            liteblocks_count += 1
            has_silu = any(isinstance(m, nn.SiLU) for m in module.pw_project.modules())
            assert not has_silu, f"Stage4中的{name}.pw_project不应该有激活函数"
    
    print(f"  找到 {liteblocks_count} 个LiteBlock")
    print(f"  ✓ 所有LiteBlock的Project层都是线性的")
    
    print("\n" + "=" * 60)
    print("✅ 所有Stage的LiteBlock配置正确")
    print("=" * 60)


if __name__ == "__main__":
    # 测试单个LiteBlock
    test_liteblock_linear_project()
    
    # 测试所有Stage中的LiteBlock
    test_all_stages_liteblock()
    
    print("\n🎯 所有测试通过！")
    print("LiteBlock的Project层确认为线性输出（仅BN，无激活）")