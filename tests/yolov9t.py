# ultralytics yolov9t를 로드해서 sequential 모델로 변환
import torch
from modules import *
from collections import OrderedDict


class YOLOv9t_Seq(nn.Module):
    """YOLOv9-t model implementation"""
    def __init__(self, ch=3, nc=80):
        super().__init__()
        self.nc = nc
        
        # Sequential model with all layers
        self.model = nn.Sequential(
            Conv(ch, 16, 3, 2),              # 0-P1/2
            Conv(16, 32, 3, 2),              # 1-P2/4
            ELAN1(32, 32, 32, 16),           # 2
            AConv(32, 64),                   # 3-P3/8
            RepNCSPELAN4(64, 64, 64, 32, 3), # 4
            AConv(64, 96),                   # 5-P4/16
            RepNCSPELAN4(96, 96, 96, 48, 3), # 6
            AConv(96, 128),                  # 7-P5/32
            RepNCSPELAN4(128, 128, 128, 64, 3), # 8
            SPPELAN(128, 128, 64),           # 9
            
            nn.Upsample(None, 2, 'nearest'), # 10
            Concat(1),                       # 11-cat backbone P4(5)
            RepNCSPELAN4(224, 96, 96, 48, 3), # 12
            nn.Upsample(None, 2, 'nearest'), # 13
            Concat(1),                       # 14 cat backbone P3(3)
            RepNCSPELAN4(160, 64, 64, 32, 3), # 15 - N3 output
            AConv(64, 48),                   # 16
            Concat(1),                       # 17 cat head P4()
            RepNCSPELAN4(144, 96, 96, 48, 3), # 18 - (P4/16-medium)
            AConv(96, 64),                   # 19
            Concat(1),                       # 20 cat head P5
            RepNCSPELAN4(192, 128, 128, 64, 3), # 21 - (P5/32-large)
        )
        
        # Detection head
        self.detect = Detect(nc, (64, 96, 128))  # 22
        self.stride = 32
        # Save indices for feature extraction
        self.save_indices = [3, 6, 9, 15, 18, 21]  # P3, P4, P5, N3, N4, N5
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass with intermediate feature extraction"""
        features  = {}
        # Pass through each layer and save specific outputs
        for i in range(len(self.model)):
            layer = self.model[i]

            # Handle Concat operations
            if isinstance(layer, Concat):
                if i == 11:  # Upsample(10) + P4(6)
                    x = layer([x, features[6]])
                elif i == 14:  # Upsample(13) + P3(3)  
                    x = layer([x, features[4]])
                elif i == 17:  # AConv(16) + P4(12) ← 수정!
                    x = layer([x, features[12]])  # layer 6이 아니라 12!
                elif i == 20:  # AConv(19) + P5(9) ← 여기는 맞음
                    x = layer([x, features[9]])
            else:
                x = layer(x)
            layer_name = layer.__class__.__name__
            
            # Save features needed for concat or output
            if i in [4, 6, 9, 12, 15, 18, 21]:
                features[i] = x
    
        # Multi-scale detection outputs
        outputs = [features[15], features[18], features[21]]
        

        # Detection head
        if self.training:
            return self.detect(outputs)
        else:
            inference_out, raw_outputs = self.detect(outputs)
            return inference_out, raw_outputs

    def load_ultralytics_weights(self, weight_path):
        """
        Load weights from Ultralytics YOLOv9 checkpoint with improved mapping
        
        Args:
            weight_path (str): Path to .pt file (e.g., 'yolov9t.pt')
        """
        print(f"Loading weights from {weight_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                source_model = checkpoint['model']
                if hasattr(source_model, 'state_dict'):
                    state_dict = source_model.state_dict()
                elif hasattr(source_model, 'float'):  # It's a model object
                    state_dict = source_model.float().state_dict()
                else:
                    state_dict = source_model
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()
        
        # Print source model structure for debugging
        print("\n=== Analyzing Source Model Structure ===")
        source_keys = list(state_dict.keys())
        
        # Group keys by layer
        layer_groups = {}
        for key in source_keys:
            parts = key.split('.')
            if len(parts) >= 2:
                if parts[0] == 'model':
                    layer_idx = parts[1]
                    if layer_idx not in layer_groups:
                        layer_groups[layer_idx] = []
                    layer_groups[layer_idx].append(key)
        
        # Show structure of problematic layers
        for idx in ['2', '4', '6', '8', '12', '15', '18', '21', '22']:
            if idx in layer_groups:
                print(f"\nLayer {idx} keys (showing first 5):")
                for key in layer_groups[idx][:5]:
                    print(f"  {key}")
        
        # Load weights with improved mapping
        self._load_state_dict_with_improved_mapping(state_dict)
        
        print("\nWeights loaded successfully!")

    def _load_state_dict_with_improved_mapping(self, source_state_dict):
        """Load state dict with improved key mapping including Detect head"""
        target_state_dict = self.state_dict()
        mapped_state_dict = OrderedDict()
        
        def find_source_key(target_key, source_keys):
            """Find matching source key for target key"""
            
            # Direct match
            if target_key in source_keys:
                return target_key
            
            # Pattern 1: detect.* -> model.22.*
            if target_key.startswith('detect.'):
                alt_key = target_key.replace('detect.', 'model.22.')
                if alt_key in source_keys:
                    return alt_key
            
            # Pattern 2: model.X.cv4 -> model.X.cv4 (already correct)
            if '.cv4.' in target_key and target_key in source_keys:
                return target_key
            
            # Pattern 3: Try fuzzy matching for similar keys
            target_parts = target_key.split('.')
            for source_key in source_keys:
                source_parts = source_key.split('.')
                
                # Match based on last 3 components and shape
                if len(target_parts) >= 3 and len(source_parts) >= 3:
                    target_suffix = '.'.join(target_parts[-3:])
                    source_suffix = '.'.join(source_parts[-3:])
                    
                    if target_suffix == source_suffix:
                        # Check shape compatibility
                        try:
                            if target_state_dict[target_key].shape == source_state_dict[source_key].shape:
                                return source_key
                        except:
                            pass
            
            return None
        
        source_keys = set(source_state_dict.keys())
        matched_count = 0
        missing_keys = []
        
        print("\n=== Mapping Details ===")
        detect_mapped = 0
        backbone_mapped = 0
        
        for target_key in target_state_dict.keys():
            source_key = find_source_key(target_key, source_keys)
            
            if source_key:
                # Verify shape compatibility
                if target_state_dict[target_key].shape == source_state_dict[source_key].shape:
                    mapped_state_dict[target_key] = source_state_dict[source_key]
                    matched_count += 1
                    
                    # Count detect vs backbone
                    if target_key.startswith('detect.'):
                        detect_mapped += 1
                    elif target_key.startswith('model.'):
                        backbone_mapped += 1
                else:
                    print(f"Shape mismatch: {target_key} {target_state_dict[target_key].shape} != {source_key} {source_state_dict[source_key].shape}")
                    missing_keys.append(target_key)
            else:
                missing_keys.append(target_key)
                # Debug: show what we're looking for
                if target_key.startswith('detect.'):
                    attempted_key = target_key.replace('detect.', 'model.22.')
                    if attempted_key in source_keys:
                        print(f"✓ Found mapping: {target_key} -> {attempted_key}")
                    else:
                        print(f"✗ Missing: {target_key} (tried: {attempted_key})")
        
        # Load mapped weights
        self.load_state_dict(mapped_state_dict, strict=False)
        
        print(f"\n=== Loading Results ===")
        print(f"Backbone parameters: {backbone_mapped}")
        print(f"Detect head parameters: {detect_mapped}")
        print(f"Total mapped: {matched_count}/{len(target_state_dict)}")
        print(f"Missing: {len(missing_keys)}")
        
        if missing_keys and len(missing_keys) <= 20:
            print("\nMissing keys:")
            for key in missing_keys:
                print(f"  {key}")
        
        return matched_count, missing_keys


    # 또는 더 직접적인 방법
    def load_ultralytics_weights_fixed(self, weight_path):
        """Load weights with explicit Detect head mapping"""
        print(f"Loading weights from {weight_path}...")
        
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            source_model = checkpoint['model']
            if hasattr(source_model, 'state_dict'):
                state_dict = source_model.state_dict()
            else:
                state_dict = source_model
        else:
            state_dict = checkpoint
        
        # Create new state dict with mapped keys
        new_state_dict = OrderedDict()
        
        for source_key, source_param in state_dict.items():
            # Map model.22.* to detect.*
            if source_key.startswith('model.22.'):
                target_key = source_key.replace('model.22.', 'detect.')
                new_state_dict[target_key] = source_param
            else:
                new_state_dict[source_key] = source_param
        
        # Now load with the remapped keys
        target_state = self.state_dict()
        final_state = OrderedDict()
        
        matched = 0
        missing = []
        
        for target_key in target_state.keys():
            if target_key in new_state_dict:
                if target_state[target_key].shape == new_state_dict[target_key].shape:
                    final_state[target_key] = new_state_dict[target_key]
                    matched += 1
                else:
                    print(f"Shape mismatch: {target_key}")
                    missing.append(target_key)
            else:
                missing.append(target_key)
        
        self.load_state_dict(final_state, strict=False)
        
        print(f"\n✓ Loaded: {matched}/{len(target_state)} parameters")
        print(f"✗ Missing: {len(missing)} parameters")
        
        if missing:
            print("\nMissing keys (first 10):")
            for key in missing[:10]:
                print(f"  {key}")
        
        print("\nWeights loaded!")

# 실행해서 구조 확인
def load_yolov9t():
    
    """Load custom YOLOv9t with all weights including Detect head"""
    from collections import OrderedDict
    
    # Load Ultralytics model
    checkpoint = torch.load('yolov9t.pt', map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        source_model = checkpoint['model']
        if hasattr(source_model, 'state_dict'):
            source_state = source_model.state_dict()
        else:
            source_state = source_model
    else:
        source_state = checkpoint
    
    # Create custom model
    model = YOLOv9t_Seq(nc=80, ch=3)
    target_state = model.state_dict()
    
    # Map weights with Detect head fix
    new_state = OrderedDict()
    
    for target_key in target_state.keys():
        source_key = target_key
        
        # Map detect.* to model.22.*
        if target_key.startswith('detect.'):
            source_key = target_key.replace('detect.', 'model.22.')
        
        if source_key in source_state:
            if target_state[target_key].shape == source_state[source_key].shape:
                new_state[target_key] = source_state[source_key]
                print(f"✓ Mapped: {target_key} <- {source_key}")
    
    # Load weights
    model.load_state_dict(new_state, strict=False)
    
    # Save with correct mapping
    torch.save(model.state_dict(), 'yolov9t_converted.pth')
    print(f"\nSaved {len(new_state)}/{len(target_state)} parameters")
    
    return model

if __name__ == "__main__":
    model = load_yolov9t()
    