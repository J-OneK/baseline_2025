"""Nested CPU-only initial weights for controlled Transformer-depth studies."""

import hashlib
import json
import os
from pathlib import Path
from contextlib import contextmanager

import torch

from .data import PreparedData
from .model import SequenceRetriever


LAYER_FAMILIES = {'attention', 'ffn', 'norm_attn', 'norm_ffn'}


@contextmanager
def cpu_initialization_context():
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            yield
    finally:
        torch.set_num_threads(threads)


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def parameter_digest(tensor):
    if tensor.device.type != 'cpu':
        raise ValueError('Initialization hashing requires CPU tensors.')
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def ordered_digest(state, names):
    """Match the historical train.initialization_hash ordering and exclusions."""
    result = hashlib.sha256()
    for name in names:
        if not name.startswith('action_emb.'):
            result.update(name.encode())
            result.update(state[name].detach().contiguous().numpy().tobytes())
    return result.hexdigest()


def in_depth(name, depth):
    pieces = name.split('.')
    return not (len(pieces) > 2 and pieces[0] in LAYER_FAMILIES
                and pieces[1].isdigit() and int(pieces[1]) >= depth)


def prepare_depth_initialization(cache, reference_config_path, output, max_blocks=4, extra_seed_offset=101):
    """Reconstruct old one-layer initial weights, then append nested extra layers.

    CPU-only RNG changes live inside fork_rng(devices=[]). torch.manual_seed is
    deliberately avoided because it would also modify CUDA RNG streams.
    """
    output, reference_config_path = Path(output).resolve(), Path(reference_config_path).resolve()
    if max_blocks < 1:
        raise ValueError('max_blocks must be positive.')
    config = json.loads(reference_config_path.read_text())
    if config['blocks'] != 1 or config.get('item_id_dropout', 0.) != 0:
        raise ValueError('Require an unmasked one-layer reference.')
    data = PreparedData(cache)
    signature = data.metadata['preparation_config']
    if config.get('protocol', {}).get('preparation_config', signature) != signature:
        raise ValueError('Reference configuration uses a different cache signature.')
    architecture = dict(hidden=config['hidden'], heads=config['heads'], max_blocks=max_blocks,
                        dropout=config['dropout'], use_actions=config['variant'] == 'actions')
    sources = [Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).resolve().parents[1] / 'model.py']
    identity = dict(seed=config['seed'], reference_config=str(reference_config_path),
                    reference_config_sha256=file_digest(reference_config_path),
                    cache_signature=signature, architecture=architecture,
                    extra_seed_offset=extra_seed_offset,
                    source_sha256={str(path.resolve()): file_digest(path) for path in sources})
    manifest_path = output.with_suffix('.json')
    if output.exists() or manifest_path.exists():
        if not output.exists() or not manifest_path.exists():
            raise ValueError('Incomplete initialization output; use a new output path.')
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError('Initialization identity changed; use a new output path.')
        if manifest['file_sha256'] != file_digest(output):
            raise ValueError('Initialization weights checksum differs.')
        return manifest
    with cpu_initialization_context():
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(config['seed']).get_state())
        reference = SequenceRetriever(data, config['hidden'], 1, config['heads'], config['dropout'],
                                      use_actions=architecture['use_actions'])
        reference_state = reference.state_dict()
        names = [name for name, _ in reference.named_parameters()]
        historical_hash = ordered_digest(reference_state, names)
        if historical_hash != config['shared_initialization_sha256']:
            raise ValueError('Reconstructed initial weights do not match the historical reference.')
        reference_rng = torch.get_rng_state().clone()
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(config['seed'] + extra_seed_offset).get_state())
        composite = SequenceRetriever(data, config['hidden'], max_blocks, config['heads'], config['dropout'],
                                      use_actions=architecture['use_actions'])
        state = composite.state_dict()
        for name, value in reference_state.items():
            if name not in state or state[name].shape != value.shape:
                raise ValueError(f'Reference parameter is incompatible: {name}.')
            state[name] = value
        parameter_names = [name for name, _ in composite.named_parameters()]
        reference_sha = {name: parameter_digest(reference_state[name]) for name in names}
        composite_sha = {name: parameter_digest(state[name]) for name in parameter_names}
        depths = sorted({depth for depth in [1, 2, 4, max_blocks] if depth <= max_blocks})
        counts = {str(depth): sum(state[name].numel() for name in parameter_names if in_depth(name, depth))
                  for depth in depths}
        payload = dict(model=state, rng_state_cpu=reference_rng, reference_parameter_names=names,
                       cache_signature=signature, architecture=architecture,
                       reference_shared_initialization_sha256=historical_hash,
                       reference_parameter_sha256=reference_sha, composite_parameter_sha256=composite_sha)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, output)
    manifest = dict(identity, file=str(output), file_sha256=file_digest(output),
                    historical_shared_initialization_sha256=historical_hash,
                    reference_parameter_sha256=reference_sha, composite_parameter_sha256=composite_sha,
                    parameters_by_depth=counts)
    temporary = manifest_path.with_name(manifest_path.name + '.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(temporary, manifest_path)
    return manifest


def apply_depth_initialization(model, path, data):
    """Strictly load the depth-compatible subset, then restore baseline CPU RNG."""
    path = Path(path).resolve()
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise ValueError('Apply depth initialization before moving the model to a GPU.')
    manifest = json.loads(path.with_suffix('.json').read_text())
    actual_sha = file_digest(path)
    if manifest['file_sha256'] != actual_sha:
        raise ValueError('Initialization weights checksum differs.')
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    architecture = payload['architecture']
    if (payload['cache_signature'] != data.metadata['preparation_config']
            or manifest['cache_signature'] != payload['cache_signature']):
        raise ValueError('Initialization cache signature differs.')
    if (len(model.attention) > architecture['max_blocks'] or model.hidden != architecture['hidden']
            or any(layer.num_heads != architecture['heads'] for layer in model.attention)
            or model.use_actions != architecture['use_actions'] or model.dropout.p != architecture['dropout']
            or getattr(model, 'item_id_dropout', 0) != 0):
        raise ValueError('Initialization architecture differs.')
    expected = model.state_dict()
    state = payload['model']
    if any(name not in state or state[name].shape != value.shape or state[name].dtype != value.dtype
           for name, value in expected.items()):
        raise ValueError('Initialization parameter names, shapes or dtypes differ.')
    extra = set(state) - set(expected)
    if any(in_depth(name, len(model.attention)) for name in extra):
        raise ValueError('Initialization contains unexpected non-layer parameters.')
    selected = {name: state[name] for name in expected}
    common_hash = ordered_digest(selected, payload['reference_parameter_names'])
    if common_hash != payload['reference_shared_initialization_sha256']:
        raise ValueError('Common initialization parameters do not match the one-layer reference.')
    model.load_state_dict(selected, strict=True)
    parameter_sha = {name: parameter_digest(value) for name, value in model.named_parameters()}
    if any(parameter_sha[name] != payload['composite_parameter_sha256'][name] for name in parameter_sha):
        raise ValueError('Loaded parameter values differ from the composite template.')
    torch.set_rng_state(payload['rng_state_cpu'])
    return dict(initial_weights_sha256=actual_sha, common_initialization_sha256=common_hash,
                initialization_parameter_sha256=parameter_sha)
