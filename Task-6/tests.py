import unittest
import torch
from model import Mnist_Model

class TestMnistModel(unittest.TestCase):
    def setUp(self):
        self.model = Mnist_Model()

    def test_total_parameter_count(self):
        '''Testing if the total parameter count is lesser than 25000'''
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertTrue(total_params < 25000, "❌ FAILED: Model should have less than 25000 parameters")
        print(" ✅ Passed: Model has less than 25000 parameters")

    def test_use_of_batch_normalization(self):
        '''Testing use of Batch Normalization'''
        bn_layers = [module for module in self.model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        self.assertTrue(len(bn_layers) > 0, "❌ FAILED: Model should include Batch Normalization layers")
        print("✅ Passed: Model has Batch Normalization layers")
        
    def test_use_of_dropout(self):
        '''Testing if Dropout layers are being used'''
        dropout_layers = [module for module in self.model.modules() if isinstance(module, torch.nn.Dropout)]
        self.assertTrue(len(dropout_layers) > 0, "❌ FAILED: Model should include Dropout layers")
        print("✅ Passed: Model has Dropout layers")
        
    def test_use_of_fully_connected_or_gap(self):
        '''Testing if Linear and GAP layers are being used'''
        fc_layers = [module for module in self.model.modules() if isinstance(module, torch.nn.Linear)]
        gap_layers = [module for module in self.model.modules() if isinstance(module, torch.nn.AvgPool2d)]
        self.assertTrue(len(fc_layers) > 0 or len(gap_layers) > 0, "❌ FAILED: Model should have at least one Fully Connected or Global Average Pooling layer")
        print("✅ Passed: Model has Linear and GAP layers")

if __name__ == '__main__':
    unittest.main()
