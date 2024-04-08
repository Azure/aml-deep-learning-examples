# Diagnostic script for the environment
results = {}
try:
    import torch
    results['torch'] = torch.__version__
except ImportError:
    results['torch'] = 'Not installed'

try:
    results['torch_cuda'] = torch.cuda.is_available()
except ImportError:
    results['torch_cuda'] = 'Not installed'

try:
    import deepspeed
    results['deepspeed'] = deepspeed.__version__
except ImportError:
    results['deepspeed'] = 'Not installed'

try:
    import axolotl
    results['axolotl'] = axolotl.__version__

except ImportError:
    results['axolotl'] = 'Not installed'

try:
    import nebulaml as nm
    results['nebula'] = nm.__version__
except ImportError:
    results['nebula'] = 'Not installed'

if __name__ == '__main__':
    for k,v in results.items():
        print(f'{k}: {v}')