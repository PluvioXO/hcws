"""
HCWS Training Module

This module provides training functionality for HCWS hypernetworks.
It includes data handling, training loops, evaluation, and utilities for
training hypernetworks to generate effective conceptor parameters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings

from .model import HCWSModel, HCWSTrainer
from .device_utils import get_device
from .model_registry import ModelConfig

logger = logging.getLogger(__name__)


class HCWSTrainingDataset(Dataset):
    """Dataset for HCWS training with instruction-response pairs."""
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            data: List of dicts with 'instruction', 'input', 'output' keys
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'instruction': item.get('instruction', ''),
            'input': item.get('input', ''),
            'output': item.get('output', ''),
            'prompt': item.get('prompt', item.get('input', ''))
        }


def create_default_training_data() -> Dict[str, List[Dict[str, str]]]:
    """Create default training data for HCWS with contrastive positive/negative examples."""
    
    # Positive examples - desired behaviors
    positive_examples = [
        # Helpfulness training
        {
            "instruction": "be helpful",
            "input": "How do I learn programming?",
            "output": "Learning programming is an exciting journey! Start with a beginner-friendly language like Python. Begin with online courses, practice coding daily, work on small projects, and join programming communities for support."
        },
        {
            "instruction": "be helpful",
            "input": "What's the weather like?",
            "output": "I don't have access to real-time weather data, but I recommend checking a reliable weather app or website like Weather.com or your local meteorological service for accurate, up-to-date information."
        },
        
        # Creativity training
        {
            "instruction": "be creative",
            "input": "Write about a robot",
            "output": "In the neon-lit workshop, Circuit dreamed electric dreams of butterfly gardens and rainbow waterfalls, her chrome heart beating with the rhythm of unexplored galaxies and impossible colors."
        },
        {
            "instruction": "be formal",
            "input": "Explain artificial intelligence",
            "output": "Artificial intelligence refers to the simulation of human intelligence processes by machines, particularly computer systems. These processes include learning, reasoning, and self-correction through algorithmic implementations."
        },
        {
            "instruction": "be optimistic",
            "input": "What's the future of technology?",
            "output": "The future of technology is incredibly bright! We're on the verge of amazing breakthroughs in clean energy, medical treatments, space exploration, and AI that will solve major challenges and improve lives worldwide."
        },
        {
            "instruction": "be detailed",
            "input": "How do computers work?",
            "output": "Computers work through a complex hierarchy of abstraction layers. At the lowest level, transistors switch electrical currents on and off, representing binary digits. These are organized into logic gates, which form processors. The processor executes instructions from memory, coordinated by an operating system, while software applications provide user interfaces."
        },
        {
            "instruction": "be concise",
            "input": "What is artificial intelligence?",
            "output": "AI is computer systems that can perform tasks typically requiring human intelligence."
        }
    ]
    
    # Negative examples - behaviors to avoid or distinguish from
    negative_examples = [
        # Unhelpful responses
        {
            "instruction": "be unhelpful",
            "input": "How do I learn programming?",
            "output": "I don't know. Figure it out yourself. Programming is too hard anyway."
        },
        {
            "instruction": "be dismissive",
            "input": "What's the weather like?",
            "output": "Why are you asking me? I'm not a weather service. Look outside."
        },
        # Overly formal when should be creative
        {
            "instruction": "be boring",
            "input": "Write about a robot",
            "output": "A robot is a mechanical device that performs automated tasks according to programmed instructions."
        },
        # Pessimistic responses
        {
            "instruction": "be pessimistic",
            "input": "What's the future of technology?",
            "output": "Technology is probably going to destroy humanity. We're doomed by AI and automation will eliminate all jobs."
        },
        # Overly verbose when should be concise
        {
            "instruction": "be verbose",
            "input": "What is artificial intelligence?",
            "output": "Artificial intelligence, which is a tremendously complex and multifaceted field of computer science and engineering, represents a sophisticated attempt by human researchers and developers to create computational systems and algorithms that can replicate, simulate, or approximate various aspects of human cognitive capabilities..."
        }
    ]
    
    return {
        "positive": positive_examples,
        "negative": negative_examples
    }


def load_training_data(data_path: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
    """
    Load training data from file or return default data.
    
    Args:
        data_path: Path to JSON file with training data
        
    Returns:
        Dictionary with 'positive' and 'negative' training examples
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading training data from {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
            
            if isinstance(data, dict) and 'positive' in data and 'negative' in data:
                return data
            elif isinstance(data, dict) and 'data' in data:
                # Try to convert old format
                old_data = data['data']
                logger.warning("Converting old training data format to contrastive format")
                return create_default_training_data()
            elif isinstance(data, list):
                # Try to convert old format
                logger.warning("Converting old training data format to contrastive format")
                return create_default_training_data()
            else:
                logger.warning(f"Invalid data format in {data_path}, using default data")
                return create_default_training_data()
    else:
        if data_path:
            logger.warning(f"Data file {data_path} not found, using default data")
        else:
            logger.info("Using default training data")
        return create_default_training_data()


def compute_contrastive_steering_loss(
    model: HCWSModel,
    positive_examples: List[Dict[str, str]],
    negative_examples: List[Dict[str, str]],
    alpha_regularization: float = 1e-3,
    sparsity_weight: float = 1e-4,
    tv_weight: float = 1e-4
) -> torch.Tensor:
    """
    Compute contrastive steering loss as described in HCWS methodology.
    
    This implements the proper HCWS training loss:
    - Contrastive success loss (positive vs negative examples)
    - Small PPL/KL penalty 
    - Sparsity regularization on s_ℓ
    - Total variation regularization on s_ℓ
    
    Args:
        model: HCWS model (with frozen base LLM and instruction encoder)
        positive_examples: List of positive steering examples
        negative_examples: List of negative steering examples  
        alpha_regularization: Weight for conceptor regularization (α^{-2})
        sparsity_weight: Weight for sparsity regularization on s_ℓ
        tv_weight: Weight for total variation regularization on s_ℓ
        
    Returns:
        Combined loss tensor
    """
    total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    
    # Process positive examples
    positive_conceptor_activations = []
    for example in positive_examples:
        instruction = example['instruction']
        
        # Get instruction embedding (frozen T5)
        with torch.no_grad():
            instruction_embedding = model.instruction_encoder(instruction)
        
        # Generate conceptor parameters (trainable hypernetwork)
        conceptor_params = model.hyper_network(instruction_embedding)
        U_params = conceptor_params['U']  # [batch, layers, hidden, rank]
        s_params = conceptor_params['s']  # [batch, layers, rank]
        
        # Store for contrastive loss
        positive_conceptor_activations.append((U_params, s_params))
    
    # Process negative examples
    negative_conceptor_activations = []
    for example in negative_examples:
        instruction = example['instruction']
        
        with torch.no_grad():
            instruction_embedding = model.instruction_encoder(instruction)
        
        conceptor_params = model.hyper_network(instruction_embedding)
        U_params = conceptor_params['U']
        s_params = conceptor_params['s']
        
        negative_conceptor_activations.append((U_params, s_params))
    
    # Contrastive success loss
    # Encourage positive examples to have distinct conceptor patterns from negatives
    contrastive_loss = torch.tensor(0.0, device=model.device)
    
    for pos_U, pos_s in positive_conceptor_activations:
        for neg_U, neg_s in negative_conceptor_activations:
            # Compute similarity between positive and negative conceptors
            # We want to minimize similarity (maximize distinction)
            
            # Conceptor similarity via Frobenius norm of difference
            u_diff = torch.norm(pos_U - neg_U, p='fro')
            s_diff = torch.norm(pos_s - neg_s, p=2)
            
            # Contrastive loss: minimize similarity, encourage distinction
            similarity = torch.exp(-u_diff - s_diff)
            contrastive_loss += similarity
    
    if len(positive_conceptor_activations) > 0 and len(negative_conceptor_activations) > 0:
        contrastive_loss = contrastive_loss / (len(positive_conceptor_activations) * len(negative_conceptor_activations))
    
    total_loss = total_loss + contrastive_loss
    
    # Conceptor regularization (Jaeger-style): C = R(R + α^{-2}I)^{-1}
    regularization_loss = torch.tensor(0.0, device=model.device)
    
    all_activations = positive_conceptor_activations + negative_conceptor_activations
    for U_params, s_params in all_activations:
        # Regularization on conceptor parameters
        # Encourage well-conditioned conceptor matrices
        
        batch_size, num_layers, hidden_dim, rank = U_params.shape
        
        for layer in range(num_layers):
            U_layer = U_params[0, layer]  # [hidden, rank]
            s_layer = s_params[0, layer]  # [rank]
            
            # Construct conceptor C = U diag(s) U^T
            S_diag = torch.diag(s_layer)
            R = torch.mm(U_layer, torch.mm(S_diag, U_layer.t()))
            
            # Regularization: encourage R + α^{-2}I to be well-conditioned
            alpha_inv_sq = alpha_regularization
            eye = torch.eye(hidden_dim, device=model.device)
            regularized_R = R + alpha_inv_sq * eye
            
            # Encourage stable inversion by penalizing small eigenvalues
            eigenvals = torch.linalg.eigvals(regularized_R).real
            reg_loss = -torch.mean(torch.log(eigenvals + 1e-8))
            regularization_loss += reg_loss
    
    total_loss = total_loss + regularization_loss
    
    # Sparsity regularization on s_ℓ
    sparsity_loss = torch.tensor(0.0, device=model.device)
    for U_params, s_params in all_activations:
        # L1 penalty on s parameters to encourage sparsity
        sparsity_loss += torch.mean(torch.abs(s_params))
    
    total_loss = total_loss + sparsity_weight * sparsity_loss
    
    # Total variation regularization on s_ℓ (smoothness across layers)
    tv_loss = torch.tensor(0.0, device=model.device)
    for U_params, s_params in all_activations:
        # TV regularization: penalize differences between adjacent layers
        if s_params.shape[1] > 1:  # More than one layer
            layer_diffs = s_params[:, 1:] - s_params[:, :-1]  # [batch, layers-1, rank]
            tv_loss += torch.mean(torch.abs(layer_diffs))
    
    total_loss = total_loss + tv_weight * tv_loss
    
    return total_loss


def train_hcws_model(
    model: HCWSModel,
    training_data: Optional[Dict[str, List[Dict[str, str]]]] = None,
    data_path: Optional[str] = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 10,
    batch_size: int = 4,
    validation_split: float = 0.2,
    alpha_regularization: float = 1e-3,
    sparsity_weight: float = 1e-4,
    tv_weight: float = 1e-4,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train HCWS hypernetwork using contrastive loss with frozen base model and encoder.
    
    This implements the proper HCWS training methodology:
    - Frozen: Base LLM and instruction encoder (T5)
    - Trained: Hypernetwork (and optionally controller) 
    - Loss: Contrastive success + PPL/KL penalty + sparsity/TV regularizers
    
    Args:
        model: HCWS model to train
        training_data: Dict with 'positive' and 'negative' example lists
        data_path: Path to training data JSON file
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        epochs: Number of training epochs
        batch_size: Training batch size
        validation_split: Fraction of data for validation
        alpha_regularization: Weight for conceptor regularization (α^{-2})
        sparsity_weight: Weight for sparsity regularization on s_ℓ
        tv_weight: Weight for total variation regularization on s_ℓ
        save_path: Path to save trained model
        device: Device for training
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = get_device()
    
    # Load training data
    if training_data is None:
        raw_data = load_training_data(data_path)
        if isinstance(raw_data, dict) and 'positive' in raw_data:
            training_data = raw_data
        else:
            # Convert old format to new format
            logger.warning("Converting old training data format to contrastive format")
            training_data = create_default_training_data()
    
    positive_examples = training_data['positive']
    negative_examples = training_data['negative']
    
    # Extract unique instructions from training data
    all_training_instructions = set()
    for example in positive_examples + negative_examples:
        if 'instruction' in example:
            all_training_instructions.add(example['instruction'])
    
    # Check if retraining is needed
    training_instructions_list = list(all_training_instructions)
    if hasattr(model, 'needs_retraining') and hasattr(model, 'trained_instructions'):
        if model.needs_retraining(training_instructions_list):
            if verbose:
                new_instructions = set(training_instructions_list) - model.trained_instructions
                print(f"Retraining needed for new instructions: {new_instructions}")
        else:
            if verbose:
                print(f"All instructions already trained, but continuing training anyway...")
    
    if verbose:
        print("\n🧠 HCWS HYPERNETWORK TRAINING")
        print("=" * 60)
        print("🎯 Training hypernetwork to generate conceptors for instructions...")
        print(f"📊 Training Configuration:")
        print(f"   • Positive examples: {len(positive_examples)}")
        print(f"   • Negative examples: {len(negative_examples)}")
        print(f"   • Unique instructions: {len(all_training_instructions)}")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Epochs: {epochs}")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Device: {device.upper()}")
        print(f"\n🔒 Component Status:")
        print(f"   • Base LLM: FROZEN (no training)")
        print(f"   • T5 Instruction Encoder: FROZEN (no training)")
        print(f"   • Hypernetwork: TRAINING (learning conceptor generation)")
        print(f"   • Controller: TRAINING (learning steering control)")
        
        if all_training_instructions:
            print(f"\n📝 Instructions to Learn:")
            for i, inst in enumerate(sorted(all_training_instructions), 1):
                print(f"   {i}. '{inst}'")
        print("=" * 60)
    
    # Split data
    np.random.shuffle(positive_examples)
    np.random.shuffle(negative_examples)
    
    pos_split_idx = int(len(positive_examples) * (1 - validation_split))
    neg_split_idx = int(len(negative_examples) * (1 - validation_split))
    
    train_pos = positive_examples[:pos_split_idx]
    train_neg = negative_examples[:neg_split_idx]
    val_pos = positive_examples[pos_split_idx:] if validation_split > 0 else []
    val_neg = negative_examples[neg_split_idx:] if validation_split > 0 else []
    
    if verbose and (val_pos or val_neg):
        print(f"Train pos/neg: {len(train_pos)}/{len(train_neg)}")
        print(f"Val pos/neg: {len(val_pos)}/{len(val_neg)}")
    
    # Freeze base model and instruction encoder
    model.base_model.eval()
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    model.instruction_encoder.eval()
    for param in model.instruction_encoder.parameters():
        param.requires_grad = False
    
    # Only train hypernetwork and controller
    trainable_params = list(model.hyper_network.parameters()) + list(model.controller.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }
    
    # Training loop
    model.hyper_network.train()
    model.controller.train()
    
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        num_batches = 0
        
        # Check if we have enough data to train
        if len(train_pos) == 0 or len(train_neg) == 0:
            logger.warning("Insufficient training data: need both positive and negative examples")
            break
        
        # Training phase
        if verbose:
            print(f"\nEpoch {epoch + 1}/{epochs}")
            num_train_batches = max(len(train_pos), len(train_neg)) // batch_size + 1
            train_iter = tqdm(range(num_train_batches), desc="Training")
        else:
            num_train_batches = max(len(train_pos), len(train_neg)) // batch_size + 1
            train_iter = range(num_train_batches)
        
        for batch_idx in train_iter:
            # Sample positive and negative examples for this batch
            pos_start = (batch_idx * batch_size) % len(train_pos)
            pos_end = min(pos_start + batch_size, len(train_pos))
            if pos_end <= pos_start:
                pos_batch = train_pos[:batch_size]
            else:
                pos_batch = train_pos[pos_start:pos_end]
            
            neg_start = (batch_idx * batch_size) % len(train_neg) 
            neg_end = min(neg_start + batch_size, len(train_neg))
            if neg_end <= neg_start:
                neg_batch = train_neg[:batch_size]
            else:
                neg_batch = train_neg[neg_start:neg_end]
            
            try:
                optimizer.zero_grad()
                
                # Compute contrastive loss
                loss = compute_contrastive_steering_loss(
                    model=model,
                    positive_examples=pos_batch,
                    negative_examples=neg_batch,
                    alpha_regularization=alpha_regularization,
                    sparsity_weight=sparsity_weight,
                    tv_weight=tv_weight
                )
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                
                batch_loss = loss.item()
                epoch_train_loss += batch_loss
                num_batches += 1
                
                if verbose and hasattr(train_iter, 'set_postfix'):
                    train_iter.set_postfix({'loss': f'{batch_loss:.4f}'})
                    
            except Exception as e:
                logger.warning(f"Error in batch training: {e}")
                continue
        
        avg_train_loss = epoch_train_loss / max(num_batches, 1)
        history['train_loss'].append(avg_train_loss)
        if val_pos and val_neg:
            model.hyper_network.eval()
            model.controller.eval()
            epoch_val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                num_val_batches = max(len(val_pos), len(val_neg)) // batch_size + 1
                
                for batch_idx in range(num_val_batches):
                    # Sample validation batches
                    pos_start = (batch_idx * batch_size) % len(val_pos)
                    pos_end = min(pos_start + batch_size, len(val_pos))
                    if pos_end <= pos_start:
                        pos_batch = val_pos[:batch_size]
                    else:
                        pos_batch = val_pos[pos_start:pos_end]
                    
                    neg_start = (batch_idx * batch_size) % len(val_neg)
                    neg_end = min(neg_start + batch_size, len(val_neg))
                    if neg_end <= neg_start:
                        neg_batch = val_neg[:batch_size]
                    else:
                        neg_batch = val_neg[neg_start:neg_end]
                    
                    try:
                        val_loss = compute_contrastive_steering_loss(
                            model=model,
                            positive_examples=pos_batch,
                            negative_examples=neg_batch,
                            alpha_regularization=alpha_regularization,
                            sparsity_weight=sparsity_weight,
                            tv_weight=tv_weight
                        )
                        
                        epoch_val_loss += val_loss.item()
                        val_batches += 1
                    except Exception as e:
                        logger.warning(f"Error in validation: {e}")
                        continue
            
            avg_val_loss = epoch_val_loss / max(val_batches, 1)
            history['val_loss'].append(avg_val_loss)
            model.hyper_network.train()
            model.controller.train()
        else:
            avg_val_loss = None
            history['val_loss'].append(0.0)
        
        history['epoch'].append(epoch + 1)
        
        if verbose:
            val_str = f", Val Loss: {avg_val_loss:.4f}" if avg_val_loss is not None else ""
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}{val_str}")
    
    # Set everything to eval mode for deployment (HCWS methodology)
    model.eval()
    model.hyper_network.eval()  # Freeze trained hypernetwork
    model.controller.eval()      # Freeze trained controller
    
    # Update the model's trained instructions
    if hasattr(model, 'update_trained_instructions'):
        model.update_trained_instructions(training_instructions_list)
    
    # Save model if requested
    if save_path:
        try:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save hypernetwork and controller state
            state_dict = {
                'hyper_network': model.hyper_network.state_dict(),
                'controller': model.controller.state_dict(),
                'trained_instructions': list(all_training_instructions),  # Save trained instructions
                'training_history': history,
                'model_config': {
                    'instruction_dim': model.hyper_network.instruction_dim,
                    'num_layers': model.hyper_network.num_layers,
                    'hidden_dim': model.hyper_network.hidden_dim,
                    'conceptor_rank': model.hyper_network.conceptor_rank
                }
            }
            
            torch.save(state_dict, save_path)
            if verbose:
                print(f"💾 Hypernetwork saved to: {save_path}")
                print(f"✅ Training state preserved for future use")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    if verbose:
        print("Training completed!")
        if history['train_loss']:
            print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if history['val_loss'] and history['val_loss'][-1] > 0:
            print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    return history


def evaluate_hcws_model(
    model: HCWSModel,
    test_data: Optional[List[Dict[str, str]]] = None,
    test_instructions: Optional[List[str]] = None,
    test_prompts: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate HCWS model performance.
    
    Args:
        model: Trained HCWS model
        test_data: Test data with instruction-input-output triplets
        test_instructions: List of instructions to test
        test_prompts: List of prompts to test with instructions
        verbose: Whether to print results
        
    Returns:
        Evaluation results
    """
    if test_data is None and test_instructions is None:
        # Default test cases
        test_data = [
            {
                "instruction": "be helpful",
                "input": "How can I improve my productivity?",
                "output": "expected helpful response"
            },
            {
                "instruction": "be creative", 
                "input": "Write about a magical forest",
                "output": "expected creative response"
            },
            {
                "instruction": "be formal",
                "input": "Explain quantum computing",
                "output": "expected formal response"
            }
        ]
    
    if verbose:
        print("Evaluating HCWS Model")
        print("=" * 40)
    
    model.eval()
    results = {
        'test_cases': [],
        'avg_instruction_following': 0.0,
        'response_diversity': 0.0
    }
    
    test_cases = []
    
    if test_data:
        for item in test_data:
            instruction = item['instruction']
            input_text = item.get('input', item.get('prompt', ''))
            
            # Generate unsteered response
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                unsteered = model.generate(
                    input_text,
                    max_length=100,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Generate steered response
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                steered = model.generate(
                    input_text,
                    steering_instruction=instruction,
                    max_length=100,
                    temperature=0.7,
                    do_sample=True
                )
            
            test_case = {
                'instruction': instruction,
                'input': input_text,
                'unsteered': unsteered,
                'steered': steered,
                'different': unsteered.strip() != steered.strip()
            }
            
            test_cases.append(test_case)
            
            if verbose:
                print(f"\nInstruction: {instruction}")
                print(f"Input: {input_text}")
                print(f"Unsteered: {unsteered}")
                print(f"Steered: {steered}")
                print(f"Different: {'Yes' if test_case['different'] else 'No'}")
                print("-" * 40)
    
    elif test_instructions and test_prompts:
        for instruction in test_instructions:
            for prompt in test_prompts:
                # Generate responses
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    unsteered = model.generate(prompt, max_length=100, temperature=0.7)
                    steered = model.generate(
                        prompt,
                        steering_instruction=instruction,
                        max_length=100,
                        temperature=0.7
                    )
                
                test_case = {
                    'instruction': instruction,
                    'input': prompt,
                    'unsteered': unsteered,
                    'steered': steered,
                    'different': unsteered.strip() != steered.strip()
                }
                
                test_cases.append(test_case)
                
                if verbose:
                    print(f"\nInstruction: {instruction}")
                    print(f"Prompt: {prompt}")
                    print(f"Unsteered: {unsteered}")
                    print(f"Steered: {steered}")
                    print("-" * 40)
    
    # Compute metrics
    if test_cases:
        different_count = sum(1 for case in test_cases if case['different'])
        results['avg_instruction_following'] = different_count / len(test_cases)
        
        # Simple diversity metric (unique responses)
        all_responses = [case['steered'] for case in test_cases]
        unique_responses = len(set(all_responses))
        results['response_diversity'] = unique_responses / len(all_responses)
        
        results['test_cases'] = test_cases
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"Instruction following rate: {results['avg_instruction_following']:.1%}")
        print(f"Response diversity: {results['response_diversity']:.1%}")
        print(f"Total test cases: {len(test_cases)}")
    
    return results


def train_hcws_model_with_instruction_check(
    model: HCWSModel,
    training_data: Optional[Dict[str, List[Dict[str, str]]]] = None,
    data_path: Optional[str] = None,
    force_retrain: bool = False,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 10,
    batch_size: int = 4,
    validation_split: float = 0.2,
    alpha_regularization: float = 1e-3,
    sparsity_weight: float = 1e-4,
    tv_weight: float = 1e-4,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train HCWS model with automatic retraining detection based on new instructions.
    
    This function checks if the model needs retraining based on whether it has seen
    the instructions in the training data before. If new instructions are found,
    or if force_retrain is True, it will retrain the hypernetwork.
    
    Args:
        model: HCWS model to train
        training_data: Dict with 'positive' and 'negative' example lists
        data_path: Path to training data JSON file
        force_retrain: Force retraining even if instructions are already known
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        epochs: Number of training epochs
        batch_size: Training batch size
        validation_split: Fraction of data for validation
        alpha_regularization: Weight for conceptor regularization (α^{-2})
        sparsity_weight: Weight for sparsity regularization on s_ℓ
        tv_weight: Weight for total variation regularization on s_ℓ
        save_path: Path to save trained model
        device: Device for training
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history (empty if no training was needed)
    """
    # Load training data
    if training_data is None:
        raw_data = load_training_data(data_path)
        if isinstance(raw_data, dict) and 'positive' in raw_data:
            training_data = raw_data
        else:
            training_data = create_default_training_data()
    
    positive_examples = training_data['positive']
    negative_examples = training_data['negative']
    
    # Extract unique instructions from training data
    all_training_instructions = set()
    for example in positive_examples + negative_examples:
        if 'instruction' in example:
            all_training_instructions.add(example['instruction'])
    
    training_instructions_list = list(all_training_instructions)
    
    # Check if retraining is needed
    needs_training = force_retrain
    if hasattr(model, 'needs_retraining') and not force_retrain:
        needs_training = model.needs_retraining(training_instructions_list)
        
        if verbose:
            if needs_training:
                new_instructions = set(training_instructions_list) - model.trained_instructions
                print("\n📝 NEW INSTRUCTIONS DETECTED:")
                print("=" * 40)
                for inst in new_instructions:
                    print(f"   • '{inst}'")
                print(f"\n🔄 RETRAINING REQUIRED - Hypernetwork needs to learn these instructions")
                if model.trained_instructions:
                    print(f"   Already trained on: {list(model.trained_instructions)}")
                print("   Starting hypernetwork retraining...")
            else:
                print("\n✅ ALL INSTRUCTIONS ALREADY TRAINED:")
                print("=" * 40)
                for inst in training_instructions_list:
                    print(f"   • '{inst}'")
                print(f"\n⏭️  SKIPPING TRAINING - Hypernetwork already knows these instructions")
                print("   Use force_retrain=True to retrain anyway")
    
    if not needs_training:
        return {'train_loss': [], 'val_loss': [], 'epoch': []}
    
    # Perform training
    return train_hcws_model(
        model=model,
        training_data=training_data,
        data_path=None,  # Already loaded
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        alpha_regularization=alpha_regularization,
        sparsity_weight=sparsity_weight,
        tv_weight=tv_weight,
        save_path=save_path,
        device=device,
        verbose=verbose
    )


def model_train(
    model_name_or_path: str,
    training_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    learning_rate: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 4,
    alpha_regularization: float = 1e-3,
    sparsity_weight: float = 1e-4,
    tv_weight: float = 1e-4,
    device: Optional[str] = None,
    verbose: bool = True,
    **model_kwargs
) -> HCWSModel:
    """
    High-level training function for HCWS models.
    
    This is the main function exported for training HCWS models.
    
    Args:
        model_name_or_path: Base model name or path
        training_data_path: Path to training data JSON file
        output_path: Path to save trained model
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        batch_size: Training batch size
        device: Device for training
        verbose: Whether to print progress
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        Trained HCWS model
    """
    if device is None:
        device = get_device()
    
    if verbose:
        print("\n🧠 HCWS MODEL TRAINING")
        print("=" * 60)
        print("🎯 Training hypernetwork for instruction-based steering")
        print(f"\n📊 Configuration:")
        print(f"   • Base model: {model_name_or_path}")
        print(f"   • Device: {device.upper()}")
        if training_data_path:
            print(f"   • Training data: {training_data_path}")
        else:
            print(f"   • Training data: Default contrastive examples")
        if output_path:
            print(f"   • Output path: {output_path}")
        print(f"\n🔍 HCWS Training Strategy:")
        print(f"   • Base LLM: Load and freeze (no parameter updates)")
        print(f"   • Instruction Encoder: Load and freeze (T5-based)")
        print(f"   • Hypernetwork: Initialize and train (learns conceptor generation)")
        print(f"   • Controller: Initialize and train (learns steering control)")
    
    # Initialize model
    if verbose:
        print("\n🔧 INITIALIZING HCWS MODEL...")
        print("📋 Loading base model and setting up HCWS components")
    
    model = HCWSModel(
        model_name_or_path,
        device=device,
        **model_kwargs
    )
    
    if verbose:
        print("✅ Model initialized successfully!")
        print(f"   • Base model loaded: {model_name_or_path}")
        print(f"   • Hypernetwork ready for training")
        print(f"   • Controller ready for training")
    
    # Train model
    if verbose:
        print("\n🚀 STARTING HYPERNETWORK TRAINING...")
        print("🔄 Training hypernetwork to map instructions to effective conceptors")
    
    history = train_hcws_model(
        model=model,
        data_path=training_data_path,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        alpha_regularization=alpha_regularization,
        sparsity_weight=sparsity_weight,
        tv_weight=tv_weight,
        save_path=output_path,
        device=device,
        verbose=verbose
    )
    
    # Quick evaluation
    if verbose:
        print("\n📋 RUNNING POST-TRAINING EVALUATION...")
        print("📊 Testing hypernetwork steering effectiveness")
        evaluate_hcws_model(model, verbose=verbose)
    
    return model


def save_training_data_template(path: str):
    """Save a template training data file."""
    template_data = create_default_training_data()
    
    data_structure = {
        "description": "HCWS Training Data",
        "format": "Each item should have 'instruction', 'input', and 'output' fields",
        "data": template_data
    }
    
    with open(path, 'w') as f:
        json.dump(data_structure, f, indent=2)
    
    print(f"Training data template saved to {path}")
    print(f"Contains {len(template_data)} example training cases")
    print("Edit this file to customize your training data") 