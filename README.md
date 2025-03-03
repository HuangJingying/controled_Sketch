
This project's aim is training a diffusion model to generate concrete sketch by inputting abstract sketch.

**Product Prototype Design**

* fine-grained control: This design enables fine-grained control over generated images by using an input image to specify key features, rather than relying on detailed textual prompts. （局部控制？但我认为局部控制这个表述不好）
* interprets the features: The model interprets the features of the input image—such as the number of windows, doors, or other elements—and generates a more complex, fully-rendered version of the image while preserving those key features. （可以理解用户输入的feature）

* For example, if the input image is a rough sketch or simple illustration of a house with two windows and a door, the model will output a detailed, realistic image of a house that retains exactly those features. Unlike style transfer, which preserves the original structure or composition, this approach allows for changes in the structure, style, and shape, while ensuring the core elements specified in the input image are maintained.

**Models**

* VQGAN (x)
* Stable Diffusion (VAE based) (+lora)
* ControlNet

**Dataset**