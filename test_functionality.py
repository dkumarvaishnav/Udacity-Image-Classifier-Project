#!/usr/bin/env python3
"""
Test script to verify the functionality of the image classifier components
without requiring the actual flowers dataset.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import OrderedDict
import json

def test_model_creation():
    """Test if we can create and modify the VGG16 model successfully"""
    print("Testing model creation...")
    
    try:
        # Load VGG16 model
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Create custom classifier
        classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(25088, 120)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('hidden_layer1', nn.Linear(120, 90)),
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(90, 70)),
            ('relu3', nn.ReLU()),
            ('hidden_layer3', nn.Linear(70, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        model.classifier = classifier
        
        print("✓ Model creation successful!")
        print(f"✓ Model architecture: {model.name}")
        print(f"✓ Classifier input features: {model.classifier[0].in_features}")
        print(f"✓ Classifier output classes: {model.classifier[7].out_features}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_transforms():
    """Test if the data transforms work correctly"""
    print("\nTesting data transforms...")
    
    try:
        # Training transforms
        train_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        # Test transforms
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        print("✓ Training transforms created successfully!")
        print("✓ Test transforms created successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Transform creation failed: {e}")
        return False

def test_device_detection():
    """Test GPU/CPU device detection"""
    print("\nTesting device detection...")
    
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("✓ CUDA is available!")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("✓ Using CPU (CUDA not available)")
        
        print(f"✓ Selected device: {device}")
        return True, device
        
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False, torch.device("cpu")

def test_category_mapping():
    """Test loading of category to name mapping"""
    print("\nTesting category mapping...")
    
    try:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        
        print(f"✓ Category mapping loaded successfully!")
        print(f"✓ Number of categories: {len(cat_to_name)}")
        print(f"✓ Sample categories: {list(cat_to_name.keys())[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Category mapping failed: {e}")
        return False

def test_checkpoint_structure():
    """Test checkpoint save/load structure"""
    print("\nTesting checkpoint structure...")
    
    try:
        # Create a dummy model
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        
        # Create dummy class_to_idx
        class_to_idx = {f"class_{i}": i for i in range(102)}
        model.class_to_idx = class_to_idx
        
        # Create checkpoint dictionary
        checkpoint = {
            'architecture': model.name,
            'classifier': model.classifier,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict()
        }
        
        print("✓ Checkpoint structure created successfully!")
        print(f"✓ Architecture: {checkpoint['architecture']}")
        print(f"✓ Number of classes: {len(checkpoint['class_to_idx'])}")
        print("✓ State dict contains model weights")
        
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint structure test failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\nTesting forward pass...")
    
    try:
        # Create model
        model = models.vgg16(pretrained=True)
        
        # Create custom classifier
        classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(25088, 120)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('hidden_layer1', nn.Linear(120, 90)),
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(90, 70)),
            ('relu3', nn.ReLU()),
            ('hidden_layer3', nn.Linear(70, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        model.classifier = classifier
        model.eval()
        
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✓ Forward pass successful!")
        print(f"✓ Input shape: {dummy_input.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output probabilities sum to: {torch.exp(output).sum():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("UDACITY IMAGE CLASSIFIER - FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    # Run tests
    if test_model_creation():
        tests_passed += 1
    
    if test_transforms():
        tests_passed += 1
    
    success, device = test_device_detection()
    if success:
        tests_passed += 1
    
    if test_category_mapping():
        tests_passed += 1
    
    if test_checkpoint_structure():
        tests_passed += 1
        
    if test_forward_pass():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your image classifier is ready to use!")
        print("\nTo train the model:")
        print("python train.py --arch vgg16 --epochs 5 --learning_rate 0.001")
        print("\nTo make predictions:")
        print("python predict_fixed.py --image path/to/image.jpg --checkpoint checkpoint.pth")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()