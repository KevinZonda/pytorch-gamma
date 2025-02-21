from Modules import *

model = ruby_pipeline(
    fork(),                                      # x           -> [x, x]
    [ Id(), seq(shape_transform_1d(1, 10)) ],    # [x, x]      -> [x, x']
    fst(lambda x: x * 2),                        # [x, x']     -> [x * 2, x']
    [activation("relu"), activation("sigmoid")], # [x * 2, x'] -> [relu(x * 2), sigmoid(x')]
    pick_at(0),                                  # [relu(x * 2), sigmoid(x')] -> relu(x * 2)
)

print(model)