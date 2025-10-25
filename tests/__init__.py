import torch


class DummyController(torch.nn.Module):
	def __init__(self, device="cpu", dtype=torch.float32):
		super().__init__()
		self.device = torch.device(device)
		self.dtype = dtype
		self.linear = torch.nn.Linear(1, 1).to(self.device, dtype=self.dtype)

	def forward(self, hidden_states, layer_idx):
		batch_size = hidden_states.size(0)
		gain = torch.ones(batch_size, 1, device=hidden_states.device, dtype=hidden_states.dtype)
		layer_weights = torch.ones(batch_size, 1, device=hidden_states.device, dtype=hidden_states.dtype)
		return gain, layer_weights


class DummyConceptor(torch.nn.Module):
	def __init__(self, device="cpu", dtype=torch.float32):
		super().__init__()
		self.device = torch.device(device)
		self.dtype = dtype

	def contract_activation(self, activation, strength):
		return activation - strength


class DummyConceptorBank(torch.nn.Module):
	def __init__(self, device="cpu", dtype=torch.float32):
		super().__init__()
		self.device = torch.device(device)
		self.dtype = dtype
		self.conceptors = torch.nn.ModuleList([DummyConceptor(device, dtype)])

	def get_conceptor(self, layer_idx):
		return self.conceptors[layer_idx]
"""
Test suite for HCWS (Hyper-Conceptor Weighted Steering).
""" 