# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os


######## Basic ########

# Folder with the BOP datasets.
if 'BOP_PATH' in os.environ:
  datasets_path = os.environ['BOP_PATH']
else:
  datasets_path = r'/home/lele/Codes/epos/datasets/carObj1'#'/home/panos/code/SPREADER_DATASET'

# Folder with pose results to be evaluated.
results_path = r'/home/lele/Codes/epos/store/tf_models/extend/infer' #'/path/to/folder/with/results'

# Folder for the calculated pose errors and performance scores.
eval_path = r'/home/lele/Codes/epos/store/tf_models/extend/eval' #'/path/to/eval/folder'

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/home/lele/Codes/epos/datasets/carObj1/VIS'#'/home/panos/code/SPREADER_DATASET/VIS'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/home/lele/Codes/epos/datasets/carObj1/VIS'#'/home/panos/code/SPREADER_DATASET/VIS'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'
