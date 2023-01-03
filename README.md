# Diffusion Probabilistic Models for Scene-Scale 3D Categorical Data

📌[Paper](http://arxiv.org/abs/2301.00527)        📌[Model](https://drive.google.com/drive/folders/1iqfql5PjKIMn0a9ucnvud3Q9JdPxKZV3?usp=sharing)

<img src=https://user-images.githubusercontent.com/65997635/210204837-3821cb37-6213-4edd-807d-88a8553e224a.png width="480"
  height="300">

Comparison of object-scale which is obtained by [Point-E](https://github.com/openai/point-e) and scene scale generation(ours). Our result includes multiple objects in a generated scene, while the object-scale generation crafts one object at a time.

## Abstract
In this paper, we learn a diffusion model to generate 3D data on a scene-scale. Specifically, our model crafts a 3D scene consisting of multiple objects, while recent diffusion research has focused on a single object. To realize our goal, we represent a scene with discrete class labels, i.e., categorical distribution, to assign multiple objects into semantic categories. Thus, we extend discrete diffusion models to learn scene-scale categorical distributions. In addition, we validate that a latent diffusion model can reduce computation costs for training and deploying. To the best of our knowledge, our work is the first to apply discrete and latent diffusion for 3D categorical data on a scene-scale. We further propose to perform semantic scene completion (SSC) by learning a conditional distribution using our diffusion model, where the condition is a partial observation in a sparse point cloud. In experiments, we empirically show that our diffusion models not only generate reasonable scenes, but also perform the scene completion task better than a discriminative model. 

## Result
### 3D Scene Generation
![image](https://github.com/zoomin-lee/scene-scale-diffusion/blob/main/images/3D_scene_generation.png?raw=true)

### Semantic Scene Completion
![image](https://github.com/zoomin-lee/scene-scale-diffusion/blob/main/images/semantic_scene_completion.png?raw=true)
