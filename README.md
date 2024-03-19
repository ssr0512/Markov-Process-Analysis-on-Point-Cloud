# Revisiting 3D point
3D point cloud analysis has recently garnered significant attention due to its capacity to provide more comprehensive information compared to 2D images. To confront the inherent irregular and unstructured properties of point clouds, recent research efforts have introduced numerous well-designed set abstraction blocks. However, few of them address the issues of information loss and feature mismatch during the sampling process. To address these problems, we have  explored the Markov process to revisit point clouds analysis, wherein different-scale point sets are treated as states, and information updating between these point sets is modeled as the probability transition. In the framework of Markov analysis, our encoder can be shown to effectively mitigate information loss in downsampled point sets, while our decoder can accurately recover corresponding features for the upsampled point sets. Furthermore, we introduce a difference-wise attention mechanism to specifically extract discriminative point features, focusing on informative point feature distillation within the states. Extensive experiments demonstrate that our method equipped with Markov process consistently achieves superior performance across a range of tasks including object classification, pose estimation, shape completion, part segmentation, and semantic segmentation.
