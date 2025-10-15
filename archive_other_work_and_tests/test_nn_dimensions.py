#!/usr/bin/env python3
"""Test script to verify NN dimension fixes without full simulation."""

import numpy as np

# Mock the required classes for testing
class MockData:
    def __init__(self, qpos_size, qvel_size, ctrl_size):
        self.qpos = np.random.random(qpos_size)
        self.qvel = np.random.random(qvel_size)
        self.ctrl = np.zeros(ctrl_size)

class MockModel:
    def __init__(self):
        pass

class MockCoreModule:
    def __init__(self):
        pass

def mock_quick_spawn(robot):
    # Return different sizes to simulate varying robot configurations
    qpos_size = np.random.randint(20, 40)  # Variable joint count
    qvel_size = qpos_size  # Velocity size matches position size
    ctrl_size = np.random.randint(15, 35)  # Variable control size
    
    model = MockModel()
    data = MockData(qpos_size, qvel_size, ctrl_size)
    return model, data, None

# Simplified NN class from the fixed version
class NN:
    def __init__(self, robot):
        RNG = np.random.default_rng(42)
        
        _, data, _ = mock_quick_spawn(robot)

        # Use fixed maximum sizes to avoid dimension mismatches
        self.max_input_size = 100  # Large enough for any reasonable robot
        self.hidden_size = 12
        self.max_output_size = 50   # Large enough for any reasonable robot
        
        # Store actual sizes for this robot
        self.actual_qpos_size = len(data.qpos.copy())
        self.actual_qvel_size = len(data.qvel.copy())
        self.actual_output_size = len(data.ctrl)
        
        # Use fixed sizes for network architecture
        self.input_size = self.max_input_size
        self.output_size = self.max_output_size
        
        # Add bias terms
        self.use_bias = True
        
        # Initialize weights
        self.weights = None
        self.RNG = RNG

    def random_controller(self):
        """Initialize neural network weights with fixed dimensions."""
        # Xavier initialization for better gradient flow
        w1_scale = np.sqrt(2.0 / (self.max_input_size + self.hidden_size))
        w2_scale = np.sqrt(2.0 / (self.hidden_size + self.hidden_size))
        w3_scale = np.sqrt(2.0 / (self.hidden_size + self.max_output_size))
        
        w1 = self.RNG.normal(0.0, w1_scale, size=(self.max_input_size, self.hidden_size))
        w2 = self.RNG.normal(0.0, w2_scale, size=(self.hidden_size, self.hidden_size))
        w3 = self.RNG.normal(0.0, w3_scale, size=(self.hidden_size, self.max_output_size))
        
        # Add biases if enabled
        if self.use_bias:
            b1 = self.RNG.normal(0.0, 0.1, size=self.hidden_size)
            b2 = self.RNG.normal(0.0, 0.1, size=self.hidden_size)
            b3 = self.RNG.normal(0.0, 0.1, size=self.max_output_size)
            self.weights = (w1, w2, w3, b1, b2, b3)
        else:
            self.weights = (w1, w2, w3)

    def forward(self, model, data):
        """Forward pass with fixed-size network and input padding."""
        if self.weights is None:
            self.random_controller()
            
        # Get current robot state
        pos_inputs = data.qpos
        vel_inputs = data.qvel
        
        # Normalize velocities to prevent explosion
        vel_inputs = np.tanh(vel_inputs * 0.1)
        
        # Combine position and velocity inputs
        current_inputs = np.concatenate([pos_inputs, vel_inputs])
        
        # Pad inputs to fixed size
        inputs = np.zeros(self.max_input_size)
        actual_size = min(len(current_inputs), self.max_input_size)
        inputs[:actual_size] = current_inputs[:actual_size]
        
        # Forward pass through network
        if self.use_bias and len(self.weights) == 6:
            w1, w2, w3, b1, b2, b3 = self.weights
            
            # Layer 1: input -> hidden
            layer1 = np.tanh(np.dot(inputs, w1) + b1)
            
            # Layer 2: hidden -> hidden  
            layer2 = np.tanh(np.dot(layer1, w2) + b2)
            
            # Layer 3: hidden -> output
            full_outputs = np.tanh(np.dot(layer2, w3) + b3)
        else:
            # Fallback to basic network without bias
            w1, w2, w3 = self.weights[:3]
            
            layer1 = np.tanh(np.dot(inputs, w1))
            layer2 = np.tanh(np.dot(layer1, w2))
            full_outputs = np.tanh(np.dot(layer2, w3))
        
        # Extract actual outputs for this robot
        outputs = full_outputs[:self.actual_output_size]
        
        # Scale outputs to appropriate range for joint control
        return outputs * (np.pi / 2)

def test_varying_robot_sizes():
    """Test that NN can handle robots with different joint counts."""
    print("Testing NN with varying robot configurations...")
    
    for i in range(10):
        print(f"\nTest {i+1}:")
        
        # Create mock robot
        robot = MockCoreModule()
        
        # Create NN
        nn = NN(robot)
        print(f"  Actual qpos size: {nn.actual_qpos_size}")
        print(f"  Actual qvel size: {nn.actual_qvel_size}")
        print(f"  Actual output size: {nn.actual_output_size}")
        print(f"  Network input size: {nn.input_size}")
        print(f"  Network output size: {nn.output_size}")
        
        # Test forward pass
        model, data, _ = mock_quick_spawn(robot)
        
        try:
            outputs = nn.forward(model, data)
            print(f"  Forward pass successful! Output shape: {outputs.shape}")
            print(f"  Expected output size: {nn.actual_output_size}")
            
            if len(outputs) == nn.actual_output_size:
                print("  ✅ Output size matches expected!")
            else:
                print("  ❌ Output size mismatch!")
                
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")

if __name__ == "__main__":
    test_varying_robot_sizes()
    print("\n🎉 Dimension testing complete!")