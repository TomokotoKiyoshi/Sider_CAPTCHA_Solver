# -*- coding: utf-8 -*-
"""
Configuration Loader for Loss Functions
Load and validate configuration from YAML files
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Configuration loader for loss functions and training settings
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration file
                        If None, uses default config/loss.yaml
        """
        if config_path is None:
            # Default to project root config/loss.yaml
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / 'config' / 'loss.yaml'
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Configuration dictionary
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def get_focal_loss_config(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get Focal Loss configuration
        
        Args:
            stage: Model stage (e.g., 'stage7'). If provided, uses stage-specific config
        
        Returns:
            Focal Loss configuration dictionary
        """
        if stage and 'stage_configs' in self.config:
            if stage in self.config['stage_configs']:
                stage_config = self.config['stage_configs'][stage]
                if 'focal_loss' in stage_config:
                    return stage_config['focal_loss']
        
        return self.config['focal_loss']
    
    def get_offset_loss_config(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get Offset Loss configuration
        
        Args:
            stage: Model stage (e.g., 'stage7'). If provided, uses stage-specific config
        
        Returns:
            Offset Loss configuration dictionary
        """
        if stage and 'stage_configs' in self.config:
            if stage in self.config['stage_configs']:
                stage_config = self.config['stage_configs'][stage]
                if 'offset_loss' in stage_config:
                    return stage_config['offset_loss']
        
        return self.config['offset_loss']
    
    def get_hard_negative_loss_config(self) -> Dict[str, Any]:
        """
        Get Hard Negative Loss configuration
        
        Returns:
            Hard Negative Loss configuration dictionary
        """
        return self.config['hard_negative_loss']
    
    def get_angle_loss_config(self) -> Dict[str, Any]:
        """
        Get Angle Loss configuration
        
        Returns:
            Angle Loss configuration dictionary
        """
        return self.config['angle_loss']
    
    def get_total_loss_config(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get Total Loss configuration
        
        Args:
            stage: Model stage (e.g., 'stage7'). If provided, uses stage-specific config
        
        Returns:
            Total Loss configuration dictionary
        """
        if stage and 'stage_configs' in self.config:
            if stage in self.config['stage_configs']:
                stage_config = self.config['stage_configs'][stage]
                if 'total_loss' in stage_config:
                    total_config = self.config['total_loss'].copy()
                    # Merge stage-specific weights
                    if 'weights' in stage_config['total_loss']:
                        total_config['weights'] = stage_config['total_loss']['weights']
                    return total_config
        
        return self.config['total_loss']
    
    def get_all_loss_configs(self, stage: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get all loss configurations
        
        Args:
            stage: Model stage (e.g., 'stage7'). If provided, uses stage-specific configs
        
        Returns:
            Dictionary containing all loss configurations
        """
        return {
            'focal_loss': self.get_focal_loss_config(stage),
            'offset_loss': self.get_offset_loss_config(stage),
            'hard_negative_loss': self.get_hard_negative_loss_config(),
            'angle_loss': self.get_angle_loss_config(),
            'total_loss': self.get_total_loss_config(stage)
        }
    
    def validate_config(self) -> bool:
        """
        Validate configuration for required fields
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        required_sections = ['focal_loss', 'offset_loss', 'total_loss']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate Focal Loss config
        focal_config = self.config['focal_loss']
        required_focal = ['alpha', 'beta', 'pos_threshold']
        for param in required_focal:
            if param not in focal_config:
                raise ValueError(f"Missing required parameter in focal_loss: {param}")
        
        # Validate Offset Loss config
        offset_config = self.config['offset_loss']
        required_offset = ['loss_type', 'pos_threshold']
        for param in required_offset:
            if param not in offset_config:
                raise ValueError(f"Missing required parameter in offset_loss: {param}")
        
        # Validate Total Loss config
        total_config = self.config['total_loss']
        if 'weights' not in total_config:
            raise ValueError("Missing 'weights' in total_loss config")
        
        required_weights = ['heatmap', 'offset']
        for weight in required_weights:
            if weight not in total_config['weights']:
                raise ValueError(f"Missing required weight in total_loss: {weight}")
        
        return True
    


def load_loss_config(config_path: Optional[str] = None, stage: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to load all loss configurations
    
    Args:
        config_path: Path to configuration file
        stage: Model stage (e.g., 'stage7')
    
    Returns:
        Dictionary containing all loss configurations
    """
    loader = ConfigLoader(config_path)
    loader.validate_config()
    return loader.get_all_loss_configs(stage)


def create_loss_from_config(config_path: Optional[str] = None, stage: Optional[str] = 'stage7'):
    """
    Create loss functions from configuration
    
    Args:
        config_path: Path to configuration file
        stage: Model stage
    
    Returns:
        Dictionary of loss function instances
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from focal_loss import create_focal_loss
    from offset_loss import create_offset_loss
    from hard_negative_loss import create_hard_negative_loss
    from angle_loss import create_angle_loss
    from total_loss import create_total_loss
    
    # Load configurations
    configs = load_loss_config(config_path, stage)
    
    # Create loss functions
    losses = {
        'focal_loss': create_focal_loss(configs['focal_loss']),
        'offset_loss': create_offset_loss(configs['offset_loss']),
        'hard_negative_loss': create_hard_negative_loss(configs['hard_negative_loss']),
        'angle_loss': create_angle_loss(configs['angle_loss']),
        'total_loss': create_total_loss(configs['total_loss'])
    }
    
    return losses


if __name__ == "__main__":
    # Test configuration loading
    try:
        # Load configuration
        loader = ConfigLoader()
        print("Configuration loaded successfully!")
        
        # Validate configuration
        if loader.validate_config():
            print("Configuration is valid!")
        
        # Get specific configurations
        focal_config = loader.get_focal_loss_config('stage7')
        print("\nStage7 Focal Loss Config:")
        for key, value in focal_config.items():
            print(f"  {key}: {value}")
        
        # Get all configurations
        all_configs = loader.get_all_loss_configs('stage7')
        print("\nAll loss configurations loaded:")
        for loss_name in all_configs:
            print(f"  - {loss_name}")
        
        # Test creating losses from config
        print("\nCreating loss functions from config...")
        losses = create_loss_from_config(stage='stage7')
        print("Loss functions created:")
        for name, loss_fn in losses.items():
            print(f"  - {name}: {type(loss_fn).__name__}")
        
    except Exception as e:
        print(f"Error: {e}")