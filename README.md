# Microsoft OpenHack 2018 
https://www.microsoftevents.com/profile/form/index.cfm?PKformID=0x43738390001  

OpenHack Machine Learning - Sydney  
September 4th – 6th, 2018, 8:30 -17:00  
Cliftons, Level 3, 10 Spring Street, Sydney, NSW  

OpenHack Machine Learning brings you together with other developers in a three-day event to sharpen your Machine Learning skills through a series of structured challenges to solve problems in the Computer Vision space.
 
This journey will have you exploring Data Wrangling, Cognitive Services, Azure Machine Learning Workbench and Custom Machine Learning solutions and neural networks on industry standard frameworks such as TensorFlow and CNTK. If you complete all the challenges, your adventure will end with experience deploying your solution.
 
Throughout this event you will have the opportunity to:
Work with fellow developers to leverage industry standard Machine Learning Tools and platforms to solve a series of increasingly difficult challenges
Attend and participate in Machine Learning related technical talks with Industry Subject Matter Experts
Discuss your real-world business challenges with Microsoft Experts
Connect and network with Microsoft and industry developers from across the Machine Learning Ecosystem

## members of openhack-table19 team
This is microsoft openhack table19 team repo.  
Here are members name and github id:  

Alyse: abroombroom  
Abner: zhangabner  
Heeseob: whatifif  
Nigel: nzigel  
Verlebie: verlebie  

## There are 7 challenges

Challenge 0: Setting Up Your Workspace For A Successful OpenHack  
Challenge 1 - Gearing up for Machine Learning  
Challenge 2 - The Data in Data Science  
Challenge 3: Introduction to Custom Machine Learning  
Challenge 4: Deep Learning  
Challenge 5 - Deploy to the Cloud  
Challenge 6: Detecting Safety  


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

#### Challenge 3: Introduction to Custom Machine Learning
Background
Challenge 2 set the team up for success by providing quality data. Challenge 3 will begin a journey into custom Machine Learning.

AdventureWorks wants their Data Science team, including your team members, to begin learning how and when to perform custom Machine Learning with powerful, more programmatic APIs. Becoming proficient in Machine Learning takes some time, however, beginning with a high-level API, that is stable and has good documentation, is a great start. Some of the reasons to move on to custom Machine Learning include:

To explore different algorithms to optimize what works best for the data
To create a workflow with more control over your process
To deploy a model to a specific architecture or configuration
Certain limits in bandwidth or data size for a current service have been exceeded
In this challenge, classical or traditional Machine Learning will be used. There are times when it makes sense to use, or at least begin with, traditional Machine Learning:

No need for expensive GPU
Used in ML on Edge
Simple APIs for fast prototyping
Is used in production systems
Algorithm variety
Prerequisites
Team has a setup for sharing code and working in Jupyter
Preprocessed Gear image data from Challenge 2
An installation of the Python package called scikit-learn - check if it is installed and update it to the lastest or simply install (see Hints).
Challenge
One of the most popular and well-established Python ML packages, Scikit-Learn, is often the go-to package for starting out and is not uncommon in production systems. It has a simple, intuitive API (often modeled by other packages) and is a great place to start for learning, implementing and programming traditional ML and basic neural networks in Python.

Use the team setup and team expertise to do the following tasks.

Use a non-parametric classification method (see References) to create a model to predict the class of a new gear image, training on the preprocessed 128x128x3 gear data from Challenge 2. The algorithm should be chosen from “off-the-shelf” non-parametric algorithms for classification found in the scikit-learn library.

Split the processed data into a train and test data set in order to train a machine learning model and calculate the accuracy of that model. In order to optimize model quality, the training data set is usually much larger than the test set while still leaving enough data to adequately test the model.

Find more information in the References and Hints.

Perform the following as a team:

Split the preprocessed image array data from Challenge 2 into train and test sets
Choose an algorithm from scikit-learn documentation
Train the model with the training data from the split
Predict the class of the following piece of gear with the model: here
Evaluate the model with a confusion matrix to see how individual classes performed (use test data from the split)
Output the overall accuracy (use test data from the split)
Success Criteria
The team will run one code cell in a Jupyter notebook for the coach predicting the class successfully of a piece of gear in the provided URL above.
The team will run one code cell in a Jupyter notebook for the coach showing the accuracy score on the test data from the split. This score should be above 80%.
References
Read me first

scikit-learn algorithm cheatsheet Ref
jupyter Ref
Non-parametric and parametric algorithm differences Ref
ML and Scikit-Learn

scikit-learn Machine Learning guide with vocabulary Ref
scikit-learn Supervised Learning Ref
scikit-learn General User Guide Ref
Hints
Install packages with pip install package_name if it is not installed and pip install --upgrade package_name if it needs to be updated to the latest version.
It can help to create an “image reader” function if not already built.

#### Challenge 4: Deep Learning
Background
The Data Science team at AdventureWorks and your team have now learned a great deal about classical Machine Learning and have put it to practical use. If the accuracy was acceptable, AdventureWorks would likely stick with a simple, easy-to-use frameworks like scikit-learn.

However, the gear retailer wants to try deep learning algorithms on the data to see if the accuracy of gear classification improves. AdventureWorks also anticipates an influx of catalog and outdoor pictures soon and deep learning shines when the data size gets bigger. Also, the Data Science team is eager to learn more about deep learning and Convolutional Neural Networks (CNNs) for image analysis because CNNs naturally lend themselves to performing well on complex data such as images.

What differentiates Deep Learning from the more general Artificial Neural Networks is the hidden layers in its architecture which help to better “learn” features in complex data. Neural Networks are the network architectures made up of simple neurons and Deep Learning is the implementation that can solve complex problems. Also, Deep learning solutions can require less preprocessing and feature engineering.

Prerequisites
Team decides upon a deep learning framework to use in this Challenge, after reading about several different frameworks (see References)
Team performs any installs or updates to the latest versions of frameworks in their chosen setup
Challenge
Use the team setup and expertise to do the following tasks.

NOTE: Training a Deep Learning model is faster on a GPU machine. If the team setup does not already include GPU, consider working with your coach to adjust the setup before proceeding.

Create a Convolutional Neural Network (a deep learning architecture) to classify the gear data. The architecture or design should contain the following types of layers. See References for resources and more information.

Suggested architecture:

Input Layer (3 channel image input layer)
Convolutional (2D)
Max Pooling
Convolutional (2D)
Max Pooling
Dense (Output layer)
There are plenty of examples online in the documentation of each framework to get the team started. Check the References for more.

There is architecture information in a CNTK Tutorial that is helpful in understanding these concepts and implementation.

Train a model on the training dataset using the suggested architecture or an equivalent that the team wishes to try. The team will utilize a portion of their training dataset as a validation dataset. The team may have to iterate on the architecture. Make sure the best trained model is saved to disk.

Success Criteria
Team will run a code cell in a Jupyter notebook for the coach that shows the model accuracy is 90% or greater (using the test data set from Challenge 3)
Team will show logging that demonstrates validation accuracy that reaches 90% (using the validation dataset created in this challenge)
References
Read this first

What is a convolutional neural net Ref or Video
High level overview of Machine Learning and CNNs Video
Deep Learning Frameworks

Keras (an abstraction layer that uses TensorFlow or CNTK on the backend)
Docs And Tutorials
TensorFlow
Docs And Tutorials
Suggested starting point is a CNN from a Tutorial with the Layers API
CNTK
Docs And Tutorials
Suggested starting point is a CNN from a Tutorial with the Layer API

#### Challenge 5 - Deploy to the Cloud
Background
It’s not just enough to be able to build a world class machine learning model, you need to know how to expose it so that it can be consumed by team members, developers, and 3rd parties to provide useful functionality to applications and services for AdventureWorks customers.

In this challenge you will operationalize your model as a scoring REST API in the cloud (deploy it as a webservice), so that the developers at AdventureWorks and elsewhere can then write consumer applications and use this scoring endpoint for prediction.

In this helpful diagram, the deployment phase is shown prominently as part of the Data Science lifecycle.

Data Science Lifecycle

Prerequisites
Docker engine (Docker for Windows or Docker for Mac) installed and running locally or on a VM
Azure CLI and Azure ML CLI (azure-cli and azure-cli-ml packages)
Tooling such as the curl command line tool or Postman to send a request to your model endpoint
Saved model from Challenge 3
Challenge
Use the team setup and team expertise to do the following tasks.

Deploy the team’s saved model from Challenge 3 or Challenge 4 as a real-time web service on Azure.
Use one of the following tools to deploy the model as and API to which data may be sent (e.g.Â arrays or json) and a json response received - see References for useful links.

Azure Machine Learning CLI standalone
Azure Machine Learning Workbench
Non-CLI methods, e.g.Â Flask with Docker (see a suggestion, below)
It’d be best for AdventureWorks to have a fairly simple API, so a json-serialized image would work well.

Success Criteria
Demonstrate with curl or Postman that sending an image, via a URL or serialized, to your cloud deployed web service returns the model output - the class of gear
References
Read Me First

Overview of Azure ML model management Doc
Deployment walkthrough Ref
More on Deployment

Microsoft Blog on deploying from Azure ML Workbench and the Azure ML CLI Ref
Setting up with the Azure ML CLI for deployment Doc
Non-CLI deployment methods (AML alternative) Ref
Scoring File and Schema Creation References

Example of schema generation Doc
Example of the scoring file showing a CNTK model and serializing an image as a PANDAS data type for input data to service Ref
Example of the scoring file showing a scikit-learn model and a STANDARD data type (json) for input data to service Ref
After creating a run and init methods as in the links above, plus a schema file, begin with “Register a model” found in this Doc
Note one change required: there must be, if using certain frameworks, a pip requirements file (use -p flag) when creating the manifest
Docker

Docker Docs Ref
Hints
There are different input data type options for sending up to the service and you can specify this when you generate the schema for the service call.
The team must install the Azure ML CLI into the system Python if using a DSVM and the main Python in a local setup with (from this Doc): ! sudo pip install -r https://aka.ms/az-ml-o16n-cli-requirements-file
When creating the image with the az ml cli, remember to include all files necessary with the -d flag such as the conda_dep.yml or any label files. Avoid using the -c flag for the conda_dep.yml file, using -d instead. Also, a requirements.txt file, with the pip installable packages, should be specified with the -p flag.
Cluster deployment: Refer to the rest of this Doc

#### Challenge 6: Detecting Safety
Background
AdventureWorks has partnered with a local guide service to help them collect images of mountaineers and climbers wearing helmets in an effort to encourage the use of helmets for safety. AdventureWorks wants a solution that can locate every helmet present in an image.

Object detection adds a layer of complexity and usefulness on top of classification. It predicts whether something is present, and where in the image it is located. This is important if the model needs to identify multiple instances of a class in a given image.

Prerequisites
The new helmet dataset (a list of image URLs for training and testing)
Cloud setup ! curl -O https://challenge.blob.core.windows.net/challengefiles/summit_post_urls_selected.txt
Local setup Download
Team decides upon a deep learning framework to use in this Challenge, after reading about several different frameworks (See References)
Team performs any installs or updates to the latest versions of frameworks in their chosen setup
Challenge
Obtain the images from the URL list in the given file in a format conducive to the framework of choice.

Using the deep learning framework, create an object detection solution utilizing a modern model like Faster R-CNN. This model should be able to detect and create a bounding box around each helmet present in an image.

Success Criteria
Demonstrate logging or TensorBoard output from your deep learning solution showing a maximum Mean Average Precision (mAP) over 80%
References
Read this first

What is object detection Ref
What is mAP Ref
Tooling

Visual Object Tagging Tool VoTT. Works for TensorFlow and CNTK Ref
When on Linux, the Tensorflow Object Detection API Ref
CNTK Documentation Ref
Hints
When the team is planning for image processing, explore what functionality your deep learning framework offers
