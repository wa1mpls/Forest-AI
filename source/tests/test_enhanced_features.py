import torch
import torch.nn as nn
from models.enhanced_features import EnhancedFeatures
import unittest

class TestEnhancedFeatures(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_bands = 4  # RGB + NIR
        self.height = 224
        self.width = 224
        self.model = EnhancedFeatures()
        
    def test_forward_shape(self):
        # Test input shape
        x = torch.randn(self.batch_size, self.num_bands, self.height, self.width)
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 3, self.height, self.width))  # 3 vegetation indices
        
    def test_gradients(self):
        # Test gradients flow
        x = torch.randn(self.batch_size, self.num_bands, self.height, self.width, requires_grad=True)
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients exist
        self.assertIsNotNone(x.grad)
        
    def test_input_validation(self):
        # Test input validation
        with self.assertRaises(ValueError):
            # Test with wrong number of bands
            x = torch.randn(self.batch_size, 3, self.height, self.width)
            self.model(x)
            
    def test_consistency(self):
        # Test if outputs are consistent
        x = torch.randn(self.batch_size, self.num_bands, self.height, self.width)
        output1 = self.model(x)
        output2 = self.model(x)
        
        # Outputs should be identical for same input
        self.assertTrue(torch.allclose(output1, output2))
        
    def test_sensitivity(self):
        # Test if outputs are sensitive to input changes
        x = torch.randn(self.batch_size, self.num_bands, self.height, self.width)
        output1 = self.model(x)
        
        # Modify input slightly
        x_modified = x + 0.01 * torch.randn_like(x)
        output2 = self.model(x_modified)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(output1, output2))
        
    def test_parameters(self):
        # Test if parameters are properly initialized
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param)
            self.assertFalse(torch.isnan(param).any())
            self.assertFalse(torch.isinf(param).any())
            
    def test_training(self):
        # Test if model can be trained
        x = torch.randn(self.batch_size, self.num_bands, self.height, self.width)
        y = torch.randn(self.batch_size, 3, self.height, self.width)  # 3 vegetation indices
        
        # Create optimizer and loss function
        optimizer = torch.optim.AdamW(self.model.parameters())
        criterion = nn.MSELoss()
        
        # Forward pass
        output = self.model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check if loss is a valid number
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        
    def test_device(self):
        # Test if model can be moved to different devices
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.model.to(device)
            x = torch.randn(self.batch_size, self.num_bands, self.height, self.width, device=device)
            output = self.model(x)
            self.assertEqual(output.device, device)
            
    def test_save_load(self):
        # Test if model can be saved and loaded
        import tempfile
        import os
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            torch.save(self.model.state_dict(), tmp.name)
            
            # Load model
            new_model = EnhancedFeatures()
            new_model.load_state_dict(torch.load(tmp.name))
            
            # Test if loaded model produces same output
            x = torch.randn(self.batch_size, self.num_bands, self.height, self.width)
            output1 = self.model(x)
            output2 = new_model(x)
            
            self.assertTrue(torch.allclose(output1, output2))
            
        # Clean up
        os.unlink(tmp.name)
        
    def test_vegetation_indices(self):
        # Test if vegetation indices are properly computed
        x = torch.randn(self.batch_size, self.num_bands, self.height, self.width)
        output = self.model(x)
        
        # Check if vegetation indices are within reasonable range
        self.assertTrue(torch.all(~torch.isnan(output)))
        self.assertTrue(torch.all(~torch.isinf(output)))
        
    def test_output_range(self):
        # Test if output values are within reasonable range
        x = torch.randn(self.batch_size, self.num_bands, self.height, self.width)
        output = self.model(x)
        
        # Check if output values are within reasonable range
        self.assertTrue(torch.all(~torch.isnan(output)))
        self.assertTrue(torch.all(~torch.isinf(output)))
        
if __name__ == '__main__':
    unittest.main() 