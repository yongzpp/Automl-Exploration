To Install:
1) cd nni
2) make install-dependencies
3) make build
4) make dev-install

Basic Usage for HPO:
1) Make changes as required to config.yml
2) Set up the ss.yml to declare your intentions
3) Check search.py for the supported search space and expand if required
4) Set up your search space in search_space.json
5) Run nnictl create --config ./config.yml --port **** and go to the weburl for UI

Basic Usage for NAS:
1) Set up the ss.yml to declare the architecture parameters
2) Run the nas.py file
3) If optimzation of architecture parameters is needed:
     - Update the config.yml file (change train.py to nas.py)
     - Check search.py for the supported search space and and expand if required
     - Set up your search space in search_space.json
     - Run nnictl create --config ./config.yml --port **** and go to the weburl for UI
4) To conduct HPO on trained architecture:
     - Update your preferred path in the ss.yml to reuse the preferred architecture
     - Check search.py for the supported search space and and expand if required
     - Set up your search space in search_space.json
     - Run nnictl create --config ./config.yml --port **** and go to the weburl for UI

Recreating model:
1) For sklearn and pytorch models, import and apply the weights (refer to load.py)
2) For NAS models, refer to load.py and work from there

Caveats:
1) To avoid user from specifying too many directories, leave the filenames as it is less load.py
2) Hence, also work in the same directory due to import depedencies to avoid complex user specifications
3) Pytorch do not seem to support saving of model architecture, working in NNI is required

Extensibility:
1*) Custom models and sklearn models can be added for tabular data
      - Requires fit(), predict() and score() functions
      - Import and alter the get_tabular() function in search.py
2*) Custom models and other models from pytorch can be added for image data
      - Import and alter the get_model() function in search.py
3) For custom handling of tabular data other than classification, go to preprocess_tabular() in dataset.py
* Current way of abstraction requires custom model to be under net.py file and Net() class, can be change in search.py if required

Supported Sklearn Models:
1) SVC
2) Guassian NB
3) Random Forest
4) Logistic Regression
5) K Nearest Neighbours

Supported Convolutional Models:
1) VGG
2) Resnet
3) Mobilenet

Supported Feature Extractors:
1) VGG
2) Resnet
