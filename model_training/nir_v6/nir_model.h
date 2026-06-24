// Auto-generated — NIR Hybrid CNN+SVM  (v7 RTL-faithful)
#ifndef NIR_MODEL_H
#define NIR_MODEL_H

#define NUM_CLASSES   5
#define NUM_CHANNELS  224
#define CNN_FEAT_DIM  128
#define SVM_DIM       128

#define SVM_W_SCALE   0.00001627f
#define SVM_B_SCALE   0.00007590f
#define SCALER_MEAN_SCALE  0.00006030f
#define SCALER_STD_SCALE   0.00004164f

// filter_stage.v SG-9 coefficients (÷256)
#define SG9_C0  (-23)
#define SG9_C1   (16)
#define SG9_C2   (43)
#define SG9_C3   (60)
#define SG9_C4   (65)
#define SG9_C5   (60)
#define SG9_C6   (43)
#define SG9_C7   (16)
#define SG9_C8  (-23)

#define SCALE_BACKBONE_0_WEIGHT  0.00857819f
#define SCALE_BACKBONE_1_WEIGHT  0.00005194f
#define SCALE_BACKBONE_1_BIAS  0.00003343f
#define SCALE_BACKBONE_1_RUNNING_MEAN  0.00000169f
#define SCALE_BACKBONE_1_RUNNING_VAR  0.00010820f
#define SCALE_BACKBONE_3_WEIGHT  0.00558934f
#define SCALE_BACKBONE_4_WEIGHT  0.00004416f
#define SCALE_BACKBONE_4_BIAS  0.00003040f
#define SCALE_BACKBONE_4_RUNNING_MEAN  0.00011932f
#define SCALE_BACKBONE_4_RUNNING_VAR  0.00014558f
#define SCALE_BACKBONE_7_WEIGHT  0.00380100f
#define SCALE_BACKBONE_8_WEIGHT  0.00004256f
#define SCALE_BACKBONE_8_BIAS  0.00001431f
#define SCALE_BACKBONE_8_RUNNING_MEAN  0.00009010f
#define SCALE_BACKBONE_8_RUNNING_VAR  0.00009587f
#define SCALE_HEAD_1_WEIGHT  0.00003206f
#define SCALE_HEAD_1_BIAS  0.00000601f
#define SCALE_HEAD_3_WEIGHT  0.00002283f
#define SCALE_HEAD_3_BIAS  0.00000161f

// Layer shapes
// backbone.0.weight: 64x1x25
// backbone.1.weight: 64
// backbone.1.bias: 64
// backbone.1.running_mean: 64
// backbone.1.running_var: 64
// backbone.3.weight: 128x64x5
// backbone.4.weight: 128
// backbone.4.bias: 128
// backbone.4.running_mean: 128
// backbone.4.running_var: 128
// backbone.7.weight: 128x128x3
// backbone.8.weight: 128
// backbone.8.bias: 128
// backbone.8.running_mean: 128
// backbone.8.running_var: 128
// head.1.weight: 64x128
// head.1.bias: 64
// head.3.weight: 5x64
// head.3.bias: 5

#endif  // NIR_MODEL_H
