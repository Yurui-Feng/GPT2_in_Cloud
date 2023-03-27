# Deploy GPT-2 Pretrained Model to SageMaker with Inference Endpoint
Welcome to the repository that demonstrates how to deploy a GPT-2 pretrained model to AWS SageMaker and create an inference endpoint. By following the instructions in this README, you will be able to quickly and easily set up the environment, deploy the model, and interact with it. Currently, the use case is creating a simple text-generation tool given user-specified prompt. 

We will be using the `PyTorch` framework and SageMaker is tightly integrated with `PyTorch`. The model specification (e.g. embedding dimension, number of layers and attention heads) and pretrained weights are provided by the `transformers` package from [Hugging Face](https://huggingface.co/docs/transformers/model_doc/gpt2).

# How to load and interact with GPT-2 locally

Create the environment:
```
$ conda env create --file environment.yml
```

Load pretrained GPT2 model:
```
$ python save_gpt2.py
```

Inference with a prompt:
```
$ python sample_gpt.py --out_dir=out_gpt2 --device='cpu' --start="What is the answer to life, the universe, and everything?"
```

Sample output:
```
What is the answer to life, the universe, and everything? A few more questions.

Where do life go?

Here's a little hint... humans go... to worlds with resources that they don't exist in other times. Humans, the species that exists in such worlds, found this world as a waste of resources in years ago. If they believe that others also use "all resources available on that planet," then they would believe that any other intelligent species exists also. However, as noted earlier, before the First God and Eve were killed with blood in some or all groups around their creator who didn't believe that, there were unknown few that ruled large populations.
Who sent the First God (beheading each other, and the destruction of "every group," and being sent into an alternate universe where Adam and Eve were both killed by a high-velocity, massive brain dump) into that World?

This area in the New World, apparently filled with nothing but dead cells and alligators, has no "naturalized human
```
