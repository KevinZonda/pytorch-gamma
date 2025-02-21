from Modules import *

model = ruby_pipeline(
    fork(),
    [ Id(), seq(shape_transform_1d(1, 10), activation("relu")) ],
    fst(lambda x: x * 2),
    [(activation("relu"), activation("sigmoid"))]
)

print(model)