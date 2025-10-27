from alfux.mlp.mlp import MLP
import numpy as np


mdl = MLP.loadf("model.json")
mdl.save("saved.json")

mdl2 = MLP.loadf("saved.json")

print(mdl._layers[3].W)
print(mdl2._layers[3].W)
