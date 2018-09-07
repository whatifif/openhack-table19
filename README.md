# Microsoft OpenHack 2018 
2018.09.04 ~ 2018.09.06 (3 days)
Sydney, Australia

This is microsoft openhack table19 team repo.

## members of openhack-table19 team

Here are members name and github id:

Alyse: abroombroom
Abner: zhangabner
Heeseob: whatifif
Nigel: nzigel
Valerie: valerie

## There are 6 challenges
#### Challenge 0: Setting Up Your Workspace For A Successful OpenHack
Background
Before the team can begin working on Machine Learning and Data Science tasks, everyone needs to have a development environment that will work well with common Python libraries.

Challenge
Set up an environment that is conducive for Machine Learning tasks. It should include:

Python 3.5
Jupyter or JupyterHub access
pip (Python package manager)
See additional (optional) tools here
This environment can take advantage of cloud Data Science specific Azure tooling or a Local Data Science setup on your machine.

In addition, the team is encouraged to set up:

Code sharing capabilities
A mode of team communication
Data Science Virtual Machine for Linux Ubuntu CSP
Ubuntu Data Science Virtual Machine (DSVM)

This setup has been found to help the team work together in a consistent environment.

We’ve commonly found the following setup to work very well:

Ubuntu Data Science Virtual Machine (DSVM)
OS: Ubuntu
Size: DS12 v2 Standard (4 cores / 28.00 GiB RAM / 56 GiB Temporary Storage) - may show up as CSP
Region: (Ask your coach for the appropriate region for your OpenHack)
Authentication type: Password (critical to logging in)
This will also include:
Python 3.5
Jupyterhub
Setting up one DSVM for the whole group and logging in with Jupyterhub is best to foster collaboration and consistency - ask your coach about your options
See References for more guidance and help
See data download instructions here
Determine whether any optional installs should be added to team members’ environments
Local Computer Alternative to DSVM setup
Install Anaconda if you don’t have it for your system:
Installation information Here
Create an environment with Python 3.5: conda create -n py35 python=3.5
Activate the environment. Windows: activate py35 Unix/Linux: source activate py35
You will be able to pip or conda install packages into this environment as needed going forward
If not using Anaconda, but rather a system Python install, use the Python package manager (pip) to at least install Jupyter:
pip install jupyter
Install other Data Science packages as the need arises

See data download instructions here
Determine whether any optional installs should be added to team members’ environments

Success Criteria
Run 2 code cells, one with each of the following command blocks to ensure successful setup.
Code cell 1

On a Data Science Virtual Machine:

  import sys
  ! {sys.executable} -m pip freeze
  ! {sys.executable} -m pip --version
Or if on a Local Setup:

  ! pip freeze
  ! pip --version
Code cell 2

Run this Python code:

  import sys
  sys.version
References
Ubuntu DSVM
Create a Linux Data Science Virtual Machine (DSVM) and use JupyterHub to code with your team - Video or Doc
Important - Please Read

When provisioning the Ubuntu DSVM, ensure the team chooses Password as the Authentication type
It’s recommended to not use Edge, but use a different browser
The Jupyterhub is at an address that begins with https protocol
To get to the Jupyterhub, one must click through the non-private connection warnings in browser - this is expected behavior
The Jupyterhub is at port 8000 as the Video and Docs say - links above
Use the “Python 3” kernel
To install Python packages, example commands for the Ubuntu DSVM are shown in the Docs
Data Downloads
For the cloud setup, with the DSVM, a convenient way to download the data is through OS commands within a Jupyter notebook, e.g.:

! curl -O https://challenge.blob.core.windows.net/challengefiles/gear_images.zip

For the local setup, download the gear dataset by clicking here

Optional
Git Download
Azure ML CLI Install
Using installed tools
Getting started with conda Doc
Creating and activating a conda environment Ref
Connecting a Jupyter Notebook to a specific conda environment Ref

#### Challenge 1 - Gearing up for Machine Learning
Background
AdventureWorks, a major outdoor and climbing gear retailer, wants to understand customer behavior by learning more about the gear that consumers wear. As a joint project with their Data Science team, they have provided data in the form of product catalog images for your team to get started. The team is tasked with trying several Machine Learning approaches to categorize gear present in the catalog data, a first step towards their goal.

AdventureWorks would, eventually, like to use this knowledge to classify gear in images from Mt. Rainier, a popular climbing destination for mountaineers from around the world, and other destinations as well. This will give a more realistic picture of current needs of existing and future customers worldwide.

Machine Learning solutions are complex and custom models require time, expertise, data preparation, ongoing maintenance and deployment. As a first approach, pre-built solutions offer a way to create a ready-to-use solution quickly before moving on to the work it takes to build custom models.

Prerequisites
Team has a setup for sharing code and working in Jupyter
Team has access to the Gear catalog dataset (link and instructions provided in Challenge 0, Data Downloads)
Team has at least 1 Custom Vision Account. Create Here
Challenge
Use the team setup and expertise to do the following tasks.

Custom Vision Service is a tool for building custom image classifiers. It takes advantage of Transfer Learning which is when an existing Machine Learning model can be utilized to predict similar classes. This requires less data than training from scratch.

Your challenge is to create a classification model to predict whether an image is a hardshell jacket or an insulated jacket using a portion of the jacket data in the gear catalog images.

When the model is trained, call the prediction endpoint using Python and a Jupyter Notebook to predict the class of an image not used in training. This can be an image from the catalog data that was not uploaded or an image found online.

Note: Custom Vision has an easy to use User Interface for uploading images, tagging them with their class (e.g.Â insulated_jacket) and training the model.

Success Criteria
Each team member can call the team’s Custom Vision prediction endpoint from a Jupyter Notebook to predict the class of a jacket image not used in training and show a successful response
References
Read me first

Machine Learning Demystified Video
Definitions of some common Machine Learning terms Ref
Classification description Ref
Overview diagram of the Machine Learning process Docs
Jupyter and Managing Packages

jupyter Ref
On using conda or pip to install Python packages Ref
Jupyter notebook usage Ref
Calling the Prediction API

Requests is one of the most popular Python libraries to make API calls and is easy to use Ref with an example at Code Sample
Alternatively, the Custom Vision Service gives an example of calling the service with Python’s urllib library and Python3 - Prediction 2.0 Docs
Custom Vision Service

Custom Vision Service Ref
Custom Vision Service Docs Docs
Custom Vision Python SDK (Linux) Ref

#### Challenge 2 - The Data in Data Science
Background
Challenge 1 introduced an intuitive, easy-to-use system for creating and calling a Machine Learning model. But before the images can be used to train a custom model, the image data needs preprocessing to make the images consistent and comparable.

Challenge 2 will set the team up for success by creating a good quality dataset to use when building custom Machine Learning models later on. The Data Science team at AdventureWorks, with whom your team is working, can only move forward on this project when the data has been appropriately processed.

The gear product catalog images AdventureWorks has provided are currently raw, unformatted pixel data. Before the images can be used to train a custom model, the data needs to be preprocessed to create new pixel data in a normalized, clean format, so that images are comparable within the dataset.

Data processing is an often overlooked, but critical, piece of building high performing Machine Learning models. A Machine Learning model will almost always perform poorly without data that is clean and appropriately preprocessed.

Image data is essentially pixel data. However, simply dealing with pixel data is not usually what we want to do in image analysis. Some possible preprocessing steps are: contrast enhancement, rotation, translation, and feature extraction through various transformations.

Much time and thought generally goes into these crucial steps and the steps will be different depending on the data you are using and the exact problem the model should solve. Images may have glare, saturation differences, exposure differences, contrast issues, all that could make one set of pixels not comparable to the next.

Prerequisites
Team has a setup for sharing code and working in Jupyter
The Gear catalog dataset (same as used in Challenge 1)
Challenge
Use the team setup and expertise to do the following tasks.

The team will transform all the classes of the gear images into a particular format that can be used later on: 128x128x3 pixels (this means a 3-channel, 128x128 pixel square image - but please refrain from simply stretching the images).

Read more about the concepts and challenge in the References.

Perform the following:

Pick a basic color, e.g.Â white, and pad all images that do not have 1:1 aspect ratio
Reshape, without stretching, to a 128x128x3 pixel array shape
Ensure for each image that the pixel range is from 0 to 255 (inclusive or [0, 255]) which is also called “contrast stretching”.
Note: only one method is required
Save the data to disk in a format the team deems appropriate for easily reading back in (see Hints).
Take into account when saving the data that it will be used for Classification.
Consider that numpy arrays are the common currency in Machine Learning frameworks
Shown below are two ways of pixel-value stretching to be in the [0,255] range, or 0-255 inclusive (plotted with matplotlib):

Normalized or equalized image
Normalized or equalized image
Normalize or equalize to ensure that the pixels are in a range from [0,255].

Success Criteria
The team will run one code cell in a Jupyter notebook for the coach plotting the original image and then plotting the padded and pixel-value normalized or equalized image.
The team will run one code cell in a Jupyter notebook for the coach that shows the histogram of the pixel values which should be in the range of 0 to 255, inclusive ([0, 255]).
References
Read me first

Is your data ready for data science? Doc
jupyter Ref
On using conda or pip to install Python packages Ref
Useful Packages

matplotlib on dealing with images (I/O, plotting) Ref
numpy for image manipulation/processing/visualization Ref
PIL Image module for I/O and more Ref
PIL ImageOps module which has the ability to flip, rotate, equalize and perform other operations. Ref
Concepts

Feature scaling (normalization) Ref
Code samples

Pixel intensity normalization example Ref
Hints
It might be a good idea to encapsulate the image preprocessing into a function.
The numpy package is great for image manipulation
The numpy package can be used for I/O as well (Ref) - quicker than pandas I/O
Some ways to “strech” the pixel range of an image include: pixel-intensity normalizing or equalizing the image histogram. Explore stackoverflow and PIL for some ideas.
In matplotlib a pixel value of 0 for all channels appears black, 1 appears white, and 255 appears black.
In opencv images are read in a BGR whereas matplotlib reads and expects images as RGB. Conversion information can be found here
