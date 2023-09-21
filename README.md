# DATA - BASED LEAKAGE DETECTION AND UNCERTAINTY QUANTIFICATION IN THE MANUFACTURING OF LARGE - SCALE CFRP COMPONENTS
– Specialisation Project –
## Description of the Project
The production of CFRP (carbon fiber reinforced polymer) components requires that material preforms of an application-
specific geometry are consolidated by means of heat and pressure. In that process the pressure is applied through a vaccum
setup where the workpiece is covered by vacuum film and which is made airtight by sealant tape along the boundary. In
practice it occurs that the vacuum bag contains leakages which are in most cases invisible to the human eye. Nevertheless,
leakages need to be localized and patched as they can cause porosities and voids which can constitute serious defects that
render the final product unusable. The overall goal of this specialisation project is to develop a machine learning based
methodology that leverages flow rates measured at vacuum ports during the process time to localize leakages in industrial-
scale vacuum setups.
## Goals
1. Literature review: Provide a literature review on leakage detection. Consider at least [1, 2, 3, 4].
2. Data acquisition: Describe the experimental setup, the data acquisition process, the assumed connection between
leakage positions and sensor data, and potential sources of uncertainty in the setup.
3. Data preparation: Explore the data , identify suitable preprocessing steps and, if possible, find ways to augment the
data synthetically.
4. Model training: Design a neural network architecture to predict a single pair of leakage coordinates. Train candidate
models, adapt your design if necessary, and use hyperparameter tuning to identify a final configuration. Evaluate
your final model on test data.
5. Uncertainty quantification: Implement at least one existing method to quantify and visualize the uncertainty of your
model’s predictions. You can for example consider [5, 6] as candidate approaches.
6. (Optional) Multi-leakage detection: Discuss how your neural network architecture could be adapted so as to detect
multiple leakages in one shot. Implement and test your ideas if possible.
## References
[1] C. Brauer, D. Lorenz, and L. Tondji, “Group equivariant networks for leakage detection in vacuum bagging,” in 2022
30th European Signal Processing Conference (EUSIPCO), pp. 1437–1441, IEEE, 2022.
[2] A. Haschenburger, L. Onorato, M. Sujahudeen, D. Taraczky, A. Osis, A. Bracke, M. Byelov, F. Vermeulen, and E. Oost-
hoek, “Computational methods for leakage localisation in a vacuum bag using volumetric flow rate measurements:
Delft university of technology, german aerospace center,” Production Engineering, vol. 16, no. 6, pp. 823–835, 2022.
[3] A. Haschenburger, N. Menke, and J. Stüve, “Sensor-based leakage detection in vacuum bagging,” The International
Journal of Advanced Manufacturing Technology, vol. 116, no. 7-8, pp. 2413–2424, 2021.
[4] A. Haschenburger and C. Heim, “Two-stage leak detection in vacuum bags for the production of fibre-reinforced
composite components,” CEAS Aeronautical Journal, vol. 10, no. 3, pp. 885–892, 2019.
[5] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,”
in international conference on machine learning, pp. 1050–1059, PMLR, 2016.
[6] L. Oala, C. Heiß, J. Macdonald, M. März, W. Samek, and G. Kutyniok, “Interval neural networks: Uncertainty scores,”
arXiv preprint arXiv:2003.11566, 2020.
